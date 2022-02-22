import torch

config = {

    # total sentences: 629,343

    # dataset parameters
    'dataset': 'snli',
    'train_split': 0.9,
    # 'train_split': '1_instance'
    'test_split': 0.1,

    # hyperparameters
    'learning_rate': 0.01,
    'dropout_rate': 0,

    # model parameters
    'model_path': 'model_chkpt_split_snli_100.pkl', 
    'total_epochs': 100,
    'batch_size': 250,
    'batch_size_seq': 100,
    'enc_num_layers': 2,
    'dec_num_layers': 1,
    'enc_bidirectional': True,
    'dec_bidirectional': False,
    'hidden_size': 200,
    'embed_dim': 300,
    'teacher_forcing_ratio': 1,
    'weight_decay': 0.0001,
    'save_every': 5,
    'validate_every': 5,
    'clip_threshold': 50,
    'input_cols': ['text_indices'],
    'save_path': 'model_chkpt_78k_config_2',
    'split_input_cols': ['text_indices', 'svo', 'nonsvo'],

    # vocab parameters
    'PAD_token': 0,
    'EOS_token': 1,
    'SOS_token': 2,
    'UNK_token': 3,

    # runtime parameters
    'device_gcn': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    'device_split': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    # logs
    'logs_path_split': 'logs_split.txt',
    'logs_path_gcn': 'logs_gcn.txt',
    'logs_path_seq': 'logs_seq.txt',
    'logs_path_multi': 'logs_multi.txt',
    'logs_path_svo': 'logs_path_svo.txt',
    'logs_path_svo_100' : 'logs_path_svo_100.txt',
    'logs_path_2_layer': 'logs_path_2_layer.txt',
    'logs_path_config_1': 'logs_path_config_1.txt',
    'logs_path_config_2': 'logs_path_config_2.txt',
    'logs_path_config_2_78k': 'logs_path_config_2_78k.txt'


}