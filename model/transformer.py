import copy, math, torch
import torch.nn as nn
from collections import namedtuple



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != config.hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, config.hidden_dim)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)

        if self.use_fc_layer:
            return self.dropout(self.fc(out))
        return self.dropout(out)




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.embeddings = Embeddings(config)
        layer = nn.TransformerEncoderLayer(d_model=config.hidden_dim,
                                           nhead=config.n_heads,
                                           dim_feedforward=config.pff_dim,
                                           dropout=config.dropout_ratio,
                                           activation='gelu',
                                           batch_first=True)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, src_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.embeddings = Embeddings(config)
        layer = nn.TransformerDecoderLayer(d_model=config.hidden_dim,
                                           nhead=config.n_heads,
                                           dim_feedforward=config.pff_dim,
                                           dropout=config.dropout_ratio,
                                           activation='gelu',
                                           batch_first=True)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return x



class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

        
        
    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        #Masking
        src_pad_mask = src == self.pad_id
        trg_pad_mask = trg == self.pad_id
        trg_mask = generate_square_subsequent_mask(trg.size(1)).to(self.device)
        
        #Actual Processing
        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        dec_out = self.decoder(trg, memory, tgt_mask=trg_mask, 
                               tgt_key_padding_mask=trg_pad_mask, 
                               memory_key_padding_mask=src_pad_mask)
        logit = self.generator(dec_out)
        

        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))
        
        return self.out
