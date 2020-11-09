'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class GPT2_Wrapper(nn.Module):
    def __init__(self, args):
        super(GPT2_Wrapper, self).__init__()
        config = GPT2Config(args.dict_size, args.n_positions, args.n_ctx, args.emb_size, args.n_layers,args.n_heads,)
        self.gpt = GPT2LMHeadModel(config)

    def forward(self, input_ids,lm_labels=None):
        if lm_labels is not None:
            loss, _, _ = self.gpt(input_ids,labels=lm_labels)
            return loss
        else:
            logits = self.gpt(input_ids)
            return logits

    def save(self, name, scheduler, optimizer, args, epoch, loss):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'args': args,
            'epoch': epoch,
            'loss': loss
        }, name)
