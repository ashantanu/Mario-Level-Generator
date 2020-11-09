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

    def save(self, name, optimizer, args, epoch, loss):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'args': args,
            'epoch': epoch,
            'loss': loss
        }, name)
