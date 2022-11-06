import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.data = self.read_dataset(split)


    def read_dataset(self, split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        src = self.data[idx]['src']
        trg = self.data[idx]['trg'][:-1]
        return src, trg 




def load_dataloader(config, split):
    global pad_idx
    pad_idx = config.pad_idx

    dataset = Dataset(config, split)
    
    def _collate_fn(batch):
        src_batch, trg_batch = [], []
        
        for src, trg in batch:
            src_batch.append(torch.LongTensor(src))
            trg_batch.append(torch.LongTensor(trg))
        
        batch_size = len(src_batch)
        eos_batch = torch.LongTensor([3 for _ in range(batch_size)])
        
        src_batch = pad_sequence(src_batch, 
                                 batch_first=True, 
                                 padding_value=pad_idx)
        
        trg_batch = pad_sequence(trg_batch, 
                                 batch_first=True, 
                                 padding_value=pad_idx)
        
        trg_batch = torch.column_stack((trg_batch, eos_batch))

        return {'src': src_batch, 'trg': trg_batch}

    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=False, 
                      collate_fn=_collate_fn, 
                      num_workers=2)

