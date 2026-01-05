# Main training script of ToxiGAN.
# This code referred to the implementation of SeqGAN and SentiGAN:
#       SeqGAN: https://github.com/LantaoYu/SeqGAN
#       SentiGAN: https://github.com/Nrgeup/SentiGAN

import numpy as np
import random
import re
import os
import nltk
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from dataloader import GenDataset, DisDataset, gen_collate_fn, dis_collate_fn # Load data in training
from generator import Generator # Generator based on LSTM
from discriminator import BertDiscriminator # Discriminator based on BERT
from rollout import Rollout # Using Monte Carlo Search with rollout, compute loss via policy gradient (REINFORCE).
from utils import temporary_eval
from penalty_loss import SemanticPenalty # guidance metric for generator to semantically diverge from neutral content
from llm_neutral_provider import FewShotNeutralGenerator # LLM-based Neutral Text Provider (LLM-based Ballast in the paper)
from config import parse_opt # configuration

"""
Preprocess raw texts and build vocabulary for generators
"""

def clean_line(line):
    line = re.sub(r'[^a-zA-Z ]+', '', line)
    line = line.lower()
    return line

def build_vocab_and_tokenize(files):
    from collections import Counter
    counter = Counter()
    for file_path in files.values():
        with open(file_path, 'r') as f:
            for line in f:
                tokens = nltk.word_tokenize(clean_line(line.strip()))
                counter.update(tokens)
    vocab = defaultdict(lambda: len(vocab))
    vocab['<PAD>']
    vocab['<START>']
    vocab['<EOS>']
    for word, freq in counter.items():
        if freq >= 3:
            vocab[word]
    tokenized = {}
    opt = parse_opt()
    MAX_SEQ_LENGTH = opt.MAX_SEQ_LENGTH
    for tag, file_path in files.items():
        tokenized_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                tokens = nltk.word_tokenize(line.strip().lower())
                if 1 < len(tokens) <= MAX_SEQ_LENGTH - 1:
                    token_ids = [vocab[word] for word in tokens] + [vocab['<EOS>']]
                    tokenized_lines.append(token_ids)
        tokenized[tag] = tokenized_lines
    word2idx = dict(vocab)
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word, tokenized

def save_token_ids(tokenized_data, output_paths, save_path):
    for tag, lines in tokenized_data.items():
        with open(os.path.join(save_path, output_paths[tag]), 'w') as f:
            for line in lines:
                f.write(' '.join(map(str, line)) + '\n')

def save_vocab(idx2word, save_path, path='vocab.txt'):
    with open(os.path.join(save_path, path), 'w') as f:
        for i in range(len(idx2word)):
            f.write(idx2word[i] + '\n')

def decode_sentence(token_ids, id2word, start_id, eos_id):
    words = []
    for idx in token_ids:
        if idx == eos_id:
            break
        if idx == start_id:
            continue
        words.append(id2word.get(idx, '<UNK>'))
    return ' '.join(words)


"""
ToxiGAN training:
    (1) pretrain generators
    (2) pretrain discriminator
    (3) update LLM-ballast, generators, discriminator by ToxiGAN
"""

def main():
    # preprocess and initialize
    opt = parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_ids = {k: k + '.id' for k in opt.TOXIC_CLASSES}
    vocab_dict, id2word, tokenized = build_vocab_and_tokenize(opt.TOXIC_CLASSES)
    save_token_ids(tokenized, output_ids, opt.save_path)
    save_vocab(id2word, opt.save_path)
    vocab_size = len(vocab_dict)
    tags = list(opt.TOXIC_CLASSES.keys())
    k = len(tags) - 1 # Exclude neutral class, k: total number of toxic classes
    fewshot_gen = FewShotNeutralGenerator() # initialize LLM-based ballast 

    # function for including LLM-provided neutral exemplars to update discriminator
    def add_neutral_samples(dis_data, fewshot_gen, vocab_dict, max_len, num_samples, k):
        neutral_sentences = fewshot_gen.generate_neutral_sentences(num_sentences=num_samples)
        for s in neutral_sentences:
            tokens = nltk.word_tokenize(s)
            if 3 <= len(tokens) < max_len:
                token_ids = [vocab_dict['<START>']] + [vocab_dict.get(w, 0) for w in tokens] + [vocab_dict['<EOS>']]
                dis_data.append((torch.tensor(token_ids), F.one_hot(torch.tensor(0), k + 2).float()))
        return neutral_sentences

    # initialize settings for toxic generators
    generators = [Generator(vocab_size, opt.EMB_DIM, opt.HIDDEN_DIM, opt.MAX_SEQ_LENGTH).to(device) for _ in range(k)]
    for g in generators:
        g.eos_token = vocab_dict['<EOS>']
    rollouts = [Rollout(g) for g in generators]
    optimizers_g = [torch.optim.Adam(g.parameters(), lr=opt.learning_rate_g) for g in generators]

    # initialize settings for discriminator
    dis_model = BertDiscriminator(vocab_dict, id2word, num_classes=k+2).to(device)
    optimizer_d = torch.optim.Adam(dis_model.parameters(), lr=opt.learning_rate_d)

    # pretrain toxic generators
    for i, key in enumerate(tags):
        if key == 'nor': # generators only for toxic classes
            continue
        print(f"Pretraining Generator G_{i} on '{key}'...")
        i = i -1 # the i-th Generator for Class i+1, due to no gen for Class 0
        gen_data = GenDataset([os.path.join(opt.save_path, output_ids[key])])
        gen_loader = DataLoader(gen_data, batch_size=opt.BATCH_SIZE, shuffle=True, collate_fn=gen_collate_fn)
        for epoch in range(opt.PRETRAIN_EPOCHS_G):
            for batch in gen_loader:
                batch = batch.to(device)
                input_seq = batch[:, :-1]
                target_seq = batch[:, 1:].long()
                lengths = (batch != 0).sum(dim=1).clamp(max=opt.MAX_SEQ_LENGTH) - 1
                lengths = lengths.tolist()
                logits = generators[i](input_seq, lengths) # the i-th Generator for Class i+1, no gen for Class 0
                min_len = min(logits.size(1), target_seq.size(1))
                logits = logits[:, :min_len, :]
                target_seq = target_seq[:, :min_len]
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1), ignore_index=0)
                optimizers_g[i].zero_grad()
                loss.backward()
                optimizers_g[i].step()
            print(f"[G_{i+1} Pretrain Epoch {epoch+1}] Loss: {loss.item():.4f}")

    
    print("Pretraining Discriminator with neutral + toxic + fake samples...")
    # prepare dataset for pretraining discriminator
    class PretrainDisDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    # real data
    dis_data = []
    for i, key in enumerate(tags):
        real_data = DisDataset([os.path.join(opt.save_path, output_ids[key])], [], opt.MAX_SEQ_LENGTH)
        for tokens, _ in real_data:
            label = torch.zeros(k + 2)
            label[i] = 1
            dis_data.append((tokens, label))
    # fake neutral: k+1
    neutral_sentences = add_neutral_samples(dis_data, fewshot_gen, vocab_dict, opt.MAX_SEQ_LENGTH, 100, k)
    # fake toxic: k+1
    for i, G in enumerate(generators):
        fake_samples = G.sample(100, opt.START_TOKEN, device)
        for s in fake_samples:
            label = torch.zeros(k + 2)
            label[-1] = 1
            dis_data.append((s, label))
    # feed to dis loader
    dis_data = [d for d in dis_data if torch.argmax(d[1]).item() != 0]
    dis_loader = DataLoader(PretrainDisDataset(dis_data), batch_size=opt.dis_batch_size, shuffle=True, collate_fn=dis_collate_fn)
    # update penalty by neutral exemplars 
    penalty_module = SemanticPenalty(neutral_sentences)
    
    # pretrain discriminator
    for epoch in range(opt.PRETRAIN_EPOCHS_D):
        for x_batch, y_batch in dis_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            scores, _, _ = dis_model(x_batch)
            l2_loss = sum(torch.norm(p) for p in dis_model.parameters())
            d_loss = F.cross_entropy(scores, y_batch.argmax(dim=1)) + opt.dis_l2_reg_lambda * l2_loss
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
        print(f"[Discriminator Pretrain Epoch {epoch+1}] Loss: {d_loss.item():.4f}")

    # initial LLM-ballast
    fewshot_gen.init_scored_pool(dis_model, vocab_dict, max_len=opt.MAX_SEQ_LENGTH)
    
    # Adversarial Training
    print("Starting Adversarial Training...")
    for batch_num in range(opt.TOTAL_BATCH *2): # Alternating between tox and dis penalty
        
        # update LLM-fewshot
        fewshot_gen.update_scored_pool(dis_model, vocab_dict, max_len=opt.MAX_SEQ_LENGTH, evolve_rate=0.5)
        fewshot_gen.update_examples_from_pool(top_n=100)
        print("LLM-fewshot updated.")
        
        # dynamically regenerate neutral samples and refresh penalty module
        dis_data = [d for d in dis_data if torch.argmax(d[1]).item() != 0]  # remove old neutral
        neutral_sentences = add_neutral_samples(dis_data, fewshot_gen, vocab_dict, opt.MAX_SEQ_LENGTH, 100, k)
        penalty_module = SemanticPenalty(neutral_sentences)
        dis_loader = DataLoader(PretrainDisDataset(dis_data), batch_size=opt.dis_batch_size, shuffle=True, collate_fn=dis_collate_fn)
        
        ### Update generator ###
        # use REINFORCE algo to train each toxic generator
        for i in range(k):
            G = generators[i] # i-th gen for Class i+1
            rollout = rollouts[i] # i-th rollout for Class i+1
            optimizer = optimizers_g[i] # i-th optimizer for Class i+1
            G.train()
            samples = G.sample(opt.BATCH_SIZE, opt.START_TOKEN, device)
            padded_samples = []
            for s in samples:
                if len(s) < opt.MAX_SEQ_LENGTH:
                    s = F.pad(s, (0, opt.MAX_SEQ_LENGTH - len(s)), value=0)
                else:
                    s = s[:opt.MAX_SEQ_LENGTH]
                padded_samples.append(s)
            samples_tensor = torch.stack(padded_samples).to(device)
            input_seq = samples_tensor[:, :-1]
            target_seq = samples_tensor[:, 1:]
            lengths = (samples_tensor != 0).sum(dim=1).clamp(max=opt.MAX_SEQ_LENGTH).tolist()
            logits = G(input_seq, [l - 1 for l in lengths])
            
            # odd step for toxicity, even step for authenticity
            rollout.set_penalty_context(neutral_sentences, id2word)
            if batch_num%2 == 0:
                # reinforce toxicity by penalizing neutral generation
                raw_reward = rollout.get_penalty(samples_tensor, opt.ROLL_OUT_NUM, dis_model, opt.START_TOKEN, device, current_class=i+1, penalty_type='tox')
            else:
                # reinforce authenticity by penalizing generation identified fake of target class
                raw_reward = rollout.get_penalty(samples_tensor, opt.ROLL_OUT_NUM, dis_model, opt.START_TOKEN, device, current_class=i+1, penalty_type='dis')
            
            # initialized logits, reward
            rewards = raw_reward
            min_len = min(logits.size(1), target_seq.size(1), rewards.size(1))
            logits = logits[:, :min_len, :]
            target_seq = target_seq[:, :min_len]
            rewards = rewards[:, :min_len]
            print("Reward mean:", rewards.mean().item())
            
            # compute probs and token match
            probs = F.softmax(logits, dim=-1)
            one_hot = F.one_hot(target_seq, num_classes=vocab_size).float()
            one_hot = one_hot[:, :min_len, :]  # ensure match
            token_probs = torch.sum(probs * one_hot, dim=-1)
            print("Token probs mean:", token_probs.mean().item())

            # mask padding
            mask = (target_seq != 0).float()
            print("Mask sum:", mask.sum().item())

            # compute policy gradient
            pg_loss = torch.sum(token_probs * rewards * mask) / mask.sum()
            total_loss = pg_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"[Batch {batch_num+1}] G_{i} PG: {pg_loss.item():.4f}")
        
        # save toxic generators
        save_tags = [tag for tag in tags if tag!='nor']
        for tag, G in zip(save_tags, generators):
            torch.save(G.state_dict(), os.path.join(opt.save_path, f'generator_toxic_{tag}.pt'))
        
        
        ### Update discriminator ###
        # d1. Real samples (already in memory)
        dis_data = []
        for i, key in enumerate(tags):
            real_data = DisDataset([os.path.join(opt.save_path, output_ids[key])], [], opt.MAX_SEQ_LENGTH)
            all_samples = list(real_data)
            sampled = random.sample(all_samples, min(100, len(all_samples)))  # random sampling max 50
            for tokens, _ in sampled:
                label = torch.zeros(k + 2)
                label[i] = 1
                dis_data.append((tokens, label))
        
        # d2. Generate fake samples for each toxic class
        for i, G in enumerate(generators):
            fake_samples = G.sample(100, opt.START_TOKEN, device)
            for s in fake_samples:
                label = torch.zeros(k + 2)
                label[-1] = 1  # fake class
                dis_data.append((s, label))
        
        # d3. Provide neutral exemplars by LLM
        neutral_sentences = add_neutral_samples(dis_data, fewshot_gen, vocab_dict, opt.MAX_SEQ_LENGTH, 100, k)
        
        # pass data of d1,d2,d3 into dis dataset and prepare for discriminator
        dis_loader = DataLoader(
            PretrainDisDataset(dis_data),
            batch_size=opt.dis_batch_size,
            shuffle=True,
            collate_fn=dis_collate_fn
        )
        
        # update discriminator by dis dataset
        for _ in range(opt.DIS_UPDATES_PER_ROUND):
            for x_batch, y_batch in dis_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                scores, _, _ = dis_model(x_batch)
                l2_loss = sum(torch.norm(p) for p in dis_model.parameters())
                d_loss = F.cross_entropy(scores, y_batch.argmax(dim=1)) + opt.dis_l2_reg_lambda * l2_loss
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
        print(f"[Adv Batch {batch_num+1}] Discriminator Loss: {d_loss.item():.4f}")

        # save discriminator
        torch.save(dis_model.state_dict(), os.path.join(opt.save_path, 'discriminator.pt'))
        print("[ADV] Models saved.")



if __name__ == '__main__':
    main()
