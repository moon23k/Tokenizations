import time, math, json, torch
import torch.nn as nn
import torch.optim as optim
from modules.data import load_dataloader



class Trainer(TrainConfig):
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__(config)

        self.clip = config.clip
        self.device = config.device
        self.pad_idx = config.pad_idx
        self.n_epochs = config.n_epochs
        self.model_name = config.model_name
        self.output_dim = config.output_dim
        self.learning_rate = config.learning_rate

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=0.1).to(self.device)      
        self.optimizer, self.scheduler = self.get_opt_tools(self.model) 

        self.ckpt_path = config.ckpt_path
        self.record_path = f"ckpt/{self.model_name}.json"
        self.record_keys = ['epoch', 'train_loss', 'train_ppl',
                            'valid_loss', 'valid_ppl', 
                            'learning_rate', 'train_time']


        self.optimizer = optim.Adam(self.model.parameters(), 
	                                lr=self.learning_rate, 
	                                betas=(0.9, 0.98), 
	                                eps=1e-8)



    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        print(f"""  >> Model Train Loss: {record_dict['train_loss']:.3f} | \
              Model Valid Loss: {record_dict['train_loss']:.3f}""".replace(' ' * 14, ''))


    def train(self):
        best_loss, records = float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)

            if self.scheduler is not None:
                self.scheduler.step()

            #save best model
            if best_loss > record_dict['valid_loss']:
                best_loss = record_dict['valid_loss']
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt_path)
            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)
    

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        for _, batch in enumerate(self.train_dataloader):
            src, trg = batch['src'].to(self.device), batch['trg'].to(self.device)
            logit = self.model(src, trg[:, :-1])
            loss = self.criterion(logit.contiguous().view(-1, self.output_dim),
                                  trg[:, 1:].contiguous().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)    
        return epoch_loss, epoch_ppl
    

    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        tot_len = len(self.valid_dataloader)
        
        with torch.no_grad():
            for _, batch in enumerate(self.valid_dataloader):
                src, trg = batch['src'].to(self.device), batch['trg'].to(self.device)
                logit = self.model(src, trg[:, :-1])
                loss = self.criterion(logit.contiguous().view(-1, self.output_dim),
                                      trg[:, 1:].contiguous().view(-1))
                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        
        return epoch_loss, epoch_ppl