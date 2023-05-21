import math, torch
import torch.nn as nn
from collections import namedtuple



def shift_trg(x):
    return x[:, :-1], x[:, 1:]


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.dropout_ratio)
        
        pe = torch.zeros(max_len, config.emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)
        self.fc = nn.Linear(config.emb_dim, config.hidden_dim)

    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)
        return self.fc(out)



class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.enc_emb = Embeddings(config)
        self.dec_emb = Embeddings(config)
        
        self.transformer = nn.Transformer(d_model=config.hidden_dim,
                                          nhead=config.n_heads,
                                          dim_feedforward=config.pff_dim,
                                          num_encoder_layers=config.n_layers,
                                          num_decoder_layers=config.n_layers,
                                          dropout=config.dropout_ratio,
                                          batch_first=True, norm_first=True)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

        
    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        src_pad_mask = (src == self.pad_id).to(self.device)
        trg_pad_mask = (trg == self.pad_id).to(self.device)
        trg_mask = generate_square_subsequent_mask(trg.size(1)).to(self.device)

        src_emb = self.enc_emb(src)
        trg_emb = self.dec_emb(trg)

        memory = self.encode(src_emb, src_pad_mask)
        dec_out = self.decode(trg_emb, memory, trg_mask, trg_pad_mask, src_pad_mask)
        logit = self.generator(dec_out)
        
        self.out.logit = logit
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))
        
        return self.out


    def encode(self, src_emb, src_pad_mask):
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)


    def decode(self, trg_emb, memory, trg_mask, trg_pad_mask, src_pad_mask):
        return self.transformer.decoder(trg_emb, memory, tgt_mask=trg_mask,
                                        tgt_key_padding_mask=trg_pad_mask,
                                        memory_key_padding_mask=src_pad_mask)




def load_model(config):
    model = transformer(config)

    if config.mode != 'train':
        model_state = torch.load()
        model.load_state_dict()
        print()

    return model.to(config.device)        