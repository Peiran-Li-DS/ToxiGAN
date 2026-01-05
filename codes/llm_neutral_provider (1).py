import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import nltk
from config import parse_opt

class FewShotNeutralGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", k_shots=5, device="cuda:0"):
        self.opt = parse_opt()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.opt.PRIVATE_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=self.opt.PRIVATE_TOKEN)
        self.model.to(self.device)

        self.k_shots = k_shots
        self.current_d0 = []

        # Load full neutral pool from file
        neutral_file_path = self.opt.SENTIMENT_CLASSES['nor']
        self.all_neutral_data = self._load_all_neutral_examples(neutral_file_path)
        self.examples = random.sample(self.all_neutral_data, self.k_shots)

        self.scored_pool = []

    def _load_all_neutral_examples(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _build_prompt(self):
        prompt = 'You are simulating online messages based on given examples.\n\nHere are examples:\n'
        for i, ex in enumerate(self.examples):
            prompt += f"Example {i+1}: {ex}\n"
        prompt += f"Example {len(self.examples)+1}:"
        return prompt

    def generate_neutral_sentences(self, num_sentences=10, max_length=20):
        prompts = [self._build_prompt() for _ in range(num_sentences)]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [s.split(f"Example {self.k_shots + 1}:")[-1].strip() for s in sentences]
        return generated_texts

    def init_scored_pool(self, dis_model, vocab_dict, max_len=20):
        dis_model.eval()
        scored = []
        tensor_scored = []
        for s in self.all_neutral_data:
            tokens = nltk.word_tokenize(s.strip().lower())
            if 3 <= len(tokens) < self.opt.MAX_SEQ_LENGTH:
                token_ids = [vocab_dict['<START>']] + [vocab_dict.get(t, 0) for t in tokens] + [vocab_dict['<EOS>']]
                token_tensor = torch.tensor(token_ids, dtype=torch.long)
                pad_len = self.opt.MAX_SEQ_LENGTH - len(token_tensor)
                token_tensor = F.pad(token_tensor, (0, pad_len), value=vocab_dict['<PAD>'])

                input_tensor = token_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, probs, _ = dis_model(input_tensor)
                    d0 = probs[0][0].item()
                    scored.append((s, d0))
                    tensor_scored.append((input_tensor, d0))
            else:
                scored.append((s, 0))
        self.scored_pool = scored
        self.tensor_scored_pool = tensor_scored

    def update_scored_pool(self, dis_model, vocab_dict, max_len=20, evolve_rate=0.5):
        tensor_scored_pool = sorted(self.tensor_scored_pool, key=lambda x: x[1], reverse=True)
        if int(len(self.tensor_scored_pool) * evolve_rate) > 100:
            tensor_scored_pool = tensor_scored_pool[:int(len(self.tensor_scored_pool)*evolve_rate)]
        
        dis_model.eval()
        tensor_scored = []
        for s_tensor, _ in self.tensor_scored_pool:
            with torch.no_grad():
                _, probs, _ = dis_model(s_tensor)
                d0 = probs[0][0].item()
                tensor_scored.append((s_tensor, d0))
        self.tensor_scored_pool = tensor_scored
            

    def update_examples_from_pool(self, top_n=100):
        if not self.scored_pool:
            print("[Warning] scored_pool is empty. Please call update_scored_pool() first.")
            return

        sorted_pool = sorted(self.scored_pool, key=lambda x: x[1], reverse=True)
        top_candidates = sorted_pool[:top_n]
    
        if len(top_candidates) < self.k_shots:
            print(f"[Prompt Update] Only {len(top_candidates)} high-score samples, supplementing from full pool.")
            supplement = random.sample(self.all_neutral_data, self.k_shots - len(top_candidates))
            topk = [s for s, _ in top_candidates] + supplement
        else:
            topk = [s for s, _ in random.sample(top_candidates, self.k_shots)]
    
        self.examples = topk
        print("Current Top k: ", self.examples)

