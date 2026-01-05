import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # Private Token
    parser.add_argument('--PRIVATE_TOKEN', type=str, default='YOUR_PRIVATE_TOKEN') # Your Private HuggingFace Tokens to access related models

    # Generator hyperparameters
    parser.add_argument('--PRETRAIN_EPOCHS_G', type=int, default=400)#300
    parser.add_argument('--EMB_DIM', type=int, default=300)
    parser.add_argument('--HIDDEN_DIM', type=int, default=1024)
    parser.add_argument('--MAX_SEQ_LENGTH', type=int, default=20)
    parser.add_argument('--BATCH_SIZE', type=int, default=128) # 128
    parser.add_argument('--TOTAL_BATCH', type=int, default=20) #20 Epoch num for Adversarial Training
    parser.add_argument('--START_TOKEN', type=int, default=1)
    parser.add_argument('--ROLL_OUT_NUM', type=int, default=16)
    parser.add_argument('--learning_rate_g', type=float, default=0.0001)

    # Discriminator hyperparameters
    parser.add_argument('--PRETRAIN_EPOCHS_D', type=int, default=10)#20
    parser.add_argument('--dis_embedding_dim', type=int, default=300) # 128
    parser.add_argument('--dis_filter_sizes', type=list, default=[1,2,3,4,5,6,7,8,9,10,15])
    parser.add_argument('--dis_num_filters', type=list, default=[100,200,200,200,200,100,100,100,100,100,160])
    parser.add_argument('--dis_dropout_keep_prob', type=float, default=0.5)
    parser.add_argument('--dis_l2_reg_lambda', type=float, default=0.2)
    parser.add_argument('--dis_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate_d', type=float, default=0.0001)
    parser.add_argument('--DIS_UPDATES_PER_ROUND', type=int, default=4) #4

    # Paths
    parser.add_argument('--save_path', type=str, default='.../ToxiGAN/save/')

    ### Example for training on HateXplain ###
    """
    parser.add_argument('--dataset_path', type=str, default='.../HateXplain/')

    # Sentiment class file paths
    parser.add_argument('--SENTIMENT_CLASSES', type=dict, default={
        'nor': '.../HateXplain/ans_nor.txt',
        '1': '.../HateXplain/ans_1.txt',
        '2': '.../HateXplain/ans_2.txt',
        '3': '.../HateXplain/ans_3.txt',
        '4': '.../HateXplain/ans_4.txt',
    })
    """


    # Penalty toggle
    parser.add_argument('--use_penalty', type=bool, default=True)

    opt = parser.parse_args([])
    return opt
