import numpy as np
import sentencepiece as spm
import random, yaml, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from modules.test import Tester
from modules.train import Trainer
from modules.data import load_dataloader



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config:
    def __init__(self, args):

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)
        
        self.mode = args.mode
        self.tokenizer_type = args.tokenizer_type
        self.ckpt_path = f'ckpt/{self.tokenizer_type}.pt'
        
        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'
        self.device = torch.device(self.device_type)

    
    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def main(config):
    #Prerequisites
    set_seed()
    config = Config(args)
    model = load_model(config)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{config.tokenizer_type}/tokenizer.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')    


    if config.task == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')        
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
        
    else:
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, test_dataloader)
        tester.test()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-tokenizer_type', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.tokenizer_type in ['word', 'char', 'bpe', 'unigram']
    
    main(args)