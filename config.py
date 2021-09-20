import torch

config = {

    # dataset parameters
    'dataset': 'snli',
    'train_split': 0.9,
    'test_split': 0.1,

    # hyperparameters
    'learning_rate': 0.0001,
    'dropout_rate': 0,

    # model parameters
    'model_path': 'model_chkpt_comp_snli_2000.pkl', 
    'total_epochs': 10,
    'batch_size': 25,
    'enc_num_layers': 2,
    'dec_num_layers': 1,
    'enc_bidirectional': True,
    'dec_bidirectional': False,
    'hidden_size': 200,
    'embed_dim': 300,
    'teacher_forcing_ratio': 1,
    'weight_decay': 0.0001,
    'save_every': 10,
    'validate_every': 1,
    'clip_threshold': 50,
    'input_cols': ['text_indices'],
    'split_input_cols': ['text_indices', 'svo', 'nonsvo'],

    # vocab parameters
    'PAD_token': 0,
    'EOS_token': 1,
    'SOS_token': 2,
    'UNK_token': 3,

    # runtime parameters
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")


}