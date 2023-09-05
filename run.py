import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_model,
    Trainer, 
    Tester, 
    Generator
)




def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        #Get Config Attributes from config.yaml file
        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                if (args.model == 'small') and (group == 'base_model'):
                    continue
                if (args.model == 'base') and (group == 'small_model'):
                    continue
                for key, val in params[group].items():
                    setattr(self, key, val)


        self.mode = args.mode
        self.tokenizer_type = args.tokenizer_type.upper()
        self.path = f'{self.tokenizer_type}_{args.vocab_size}'
        self.vocab_size = int(args.vocab_size[:-1]) * 1000

        os.makedirs(f'ckpt/{self.tokenizer_type}', exist_ok=True)
        self.ckpt = f"ckpt/{self.tokenizer_type}/{self.path}.pt"
        self.tokenizer_path = f"tokenizer/{self.tokenizer_type}/{self.path}.json"

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    print(f"{config.tokenizer_type}_{str(config.vocab_size)[:2]}k Tokenizer has loaded")
    return tokenizer



def main(config):
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
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
        
    elif config.mode == 'inference':
        generator = Generator(config, model, tokenizer)
        generator.inference()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-tokenizer_type', required=True)
    parser.add_argument('-vocab_size', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.model in ['small', 'base']
    assert args.tokenizer_type.upper() in ['WL', 'WP', 'BPE', 'UNI']
    assert args.vocab_size in ['10k', '20k', '30k']
    
    main(args)