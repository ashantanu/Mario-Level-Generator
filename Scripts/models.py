import torch
import torch.nn as nn

#my implementation takes in batch first and adjust itself
class TransformerModel(nn.Module):
    def __init__(self,args):
        super(TransformerModel, self).__init__()

        self.embedding_layer = nn.Embedding(args.dict_size,args.emb_size)
        self.transformer_layer = nn.Transformer(d_model=args.emb_size,num_encoder_layers=1,num_decoder_layers=1)
        self.fully_connected = nn.Linear(args.emb_size,args.dict_size,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, source, target):
        source = torch.transpose(source,0,1).contiguous()
        target = torch.transpose(target,0,1).contiguous()
        source_embedding = self.embedding_layer(source)
        target_embedding = self.embedding_layer(target)
        transformed = self.transformer_layer(source_embedding, target_embedding)
        transformed = self.fully_connected(transformed)
        transformed = self.softmax(transformed)

        return transformed

    def get_loss(self, source, target):
        decoder_target = target[:,:-1]
        decoder_output = target[:,1:]
        decoder_output = torch.transpose(decoder_output,0,1).contiguous()
        output = self.forward(source, decoder_target)
        loss = self.loss(output.view(-1,output.shape[-1]),decoder_output.view(-1))
        return loss

    def save(self, name, optimizer, args):
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'args': args
        }, name)