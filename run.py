import numpy as np
import os, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from module.test import Tester
from module.train import Trainer
from module.data import load_dataloader
from module.model import load_model
from module.search import Search

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.tokenizer_type = args.tokenizer_type.upper()
        self.path = f'{self.tokenizer_type}_{args.vocab_size}'
        self.vocab_size = int(args.vocab_size[:-1]) * 1000

        os.makedirs(f'ckpt/{self.tokenizer_type}', exist_ok=True)
        self.ckpt = f"ckpt/{self.tokenizer_type}/{self.path}.pt"

        #Tokenizer Configs
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        #Training Configs
        self.n_epochs = 10
        self.lr = 5e-4
        self.clip = 1
        self.early_stop = True
        self.patience = 3
        self.iters_to_accumulate = 4

        if self.mode == 'test':
            self.batch_size = 1
        else:
            self.batch_size = 128

        #Model Configs
        self.emb_dim = 512
        self.hidden_dim = 512
        self.pff_dim = 1024
        self.n_heads = 8
        self.n_layers = 6
        self.dropout_ratio = 0.1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.mode == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizer(config):
    tokenizer_path = f"tokenizer/{config.tokenizer_type}/{config.path}.json"
    assert os.path.exists(tokenizer_path)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]"))
            ]
        )
    
    return tokenizer



def inference(config, model, tokenizer):
    search_module = Search(config, model)

    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #Enc Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        input_seq = tokenizer.EncodeAsIds(input_seq)
        input_seq = torch.LongTensor([input_seq])

        if config.search_method == 'beam':
            output_seq = search_module.beam_search(input_seq)
        else:
            output_seq = search_module.greedy_search(input_seq)

        output_seq = tokenizer.decode(output_seq)

        print(f"Model Out Sequence >> {output_seq}")       



def main(config):
    #Prerequisites
    set_seed()
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer(config)


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')        
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()        
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, test_dataloader)
        tester.test()
    elif config.mode == 'inference':
        inference(config, model, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-tokenizer_type', required=True)
    parser.add_argument('-vocab_size', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.tokenizer_type.upper() in ['WL', 'WP', 'BPE', 'UNI']
    assert args.vocab_size in ['5k', '10k', '15k']
    
    main(args)