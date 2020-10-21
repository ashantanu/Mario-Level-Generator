import torch
import torch.nn as nn
import math
import numpy as np
import itertools

#my implementation takes in batch first and adjust itself
class TransformerModel(nn.Module):
    def __init__(self,args):
        super(TransformerModel, self).__init__()

        self.embedding_layer = nn.Embedding(args.dict_size,args.emb_size)
        self.transformer_layer = nn.Transformer(d_model=args.emb_size,num_encoder_layers=1,num_decoder_layers=1)
        self.fully_connected = nn.Linear(args.emb_size,args.dict_size,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.pos_enc = PositionalEncoding(args.emb_size,args.column_length)

    def forward(self, source, column_heights, target):
        source = torch.transpose(source,0,1).contiguous()
        target = torch.transpose(target,0,1).contiguous()
        source_embedding = self.embedding_layer(source)
        target_embedding = self.embedding_layer(target)
        source_embedding = self.pos_enc(source_embedding)
        transformed = self.transformer_layer(source_embedding, target_embedding)
        transformed = self.fully_connected(transformed)
        transformed = self.softmax(transformed)

        return transformed

    def get_loss(self, source, column_heights, target):
        decoder_target = target[:,:-1]
        decoder_output = target[:,1:]
        decoder_output = torch.transpose(decoder_output,0,1).contiguous()
        output = self.forward(source, column_heights, decoder_target)
        loss = self.loss(output.view(-1,output.shape[-1]),decoder_output.view(-1))
        return loss

    def save(self, name, optimizer, args):
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'args': args
        }, name)


class PositionalEncoding(nn.Module):
    #TODO use column lengths given in batches to get corresponding position encoding
    def __init__(self, d_model, coulmn_length, traversal='zig-zag', dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        #position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        if traversal=='zig-zag':
            position = torch.tensor(list(range(coulmn_length))*math.ceil(max_len/coulmn_length), dtype=torch.float)[:max_len].unsqueeze(1)
        column_positions = torch.tensor(list(itertools.chain.from_iterable(itertools.repeat(i, coulmn_length) 
                                           for i in range(math.ceil(max_len/coulmn_length)))), dtype=torch.float)[:max_len].unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) + torch.sin(column_positions * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) + torch.cos(column_positions * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)