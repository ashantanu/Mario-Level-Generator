import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
import argparse
import logging

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from models import TransformerModel
from create_input import create_dataloaders

logging.basicConfig(filename='../logs/model.log',
                            filemode='w',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default = "../levels_transposed/", help = 'folder_path')
    parser.add_argument('--save_folder_path', type=str, default='../levels_prediction_textfiles/',help='Output data file path')
    parser.add_argument('--input', type=str, default='mario-1-1-edited_trans.txt',help='Input data file path')
    parser.add_argument('--checkpoint_file', type=str, default='model_checkpoint.pt',help='File path for saving model')
    #parser.add_argument('--logname', type=str, default='../logs/model.log',help='File path for saving log')

    #model args 
    parser.add_argument('--emb_size', type=int, default=512,help='Embedding size')
    parser.add_argument('--batch_size', type=int, default=8,help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,help='Number of workers')
    parser.add_argument('--num_epochs', type=int, default=2,help='Number of Epochs')
    parser.add_argument('--shuffle', type=bool, default=False,help='Shuffle dataset')
    parser.add_argument('--device', type=str, default='cpu',help='cpu or gpu/cuda')
    
    #sampling args
    parser.add_argument('--max_output_length', type=int, default=4,help='max output length')
    parser.add_argument('--max_sample_input', type=int, default=30,help='max length of input for generating sample output')
    
    #input data args
    parser.add_argument('--input_max_seq_len', type=int, default=200,help='max length of input sequence of data')
    parser.add_argument('--input_step', type=int, default=10,help='number of steps to move when sampling from input')


    args = parser.parse_args()
    hyperparamters = [args.emb_size, args.max_sample_input]
    hyperparameter_string = "_".join([str(x) for x in hyperparamters])
    args.save_name = "BestCheckpoint_"+ hyperparameter_string + ".pth"
    return args

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def train_model(model,optimizer,dataloader,args):
    loss_total = 0
    #TODO Move batching to data loader
    for iteration, (X, y) in enumerate(dataloader):
        for p in model.parameters(): p.grad = None
        X = X.to(args.device)
        y = y.to(args.device)
        model.train()
        loss = model.get_loss(X,y)
        loss.backward()
        optimizer.step()
        loss_total+=loss.data

        if iteration%100==0:
            print("Iteration step:",iteration)

    return model, optimizer, loss_total

def get_prediction(model,X,args):
    X = X[:,-args.max_sample_input:]
    Y = X.detach()
    start = torch.zeros(X.shape[0],1)
    start[:,0]=word_indices['<start>']
    Y = torch.cat((start,Y),1)
    model.eval()
    X = X.to(args.device).to(torch.long)
    Y = Y.to(args.device).to(torch.long)
    pred = model(X,Y)
    return pred[:,-1,:].detach()

def sample_outputs_from_model(test_dataloader,args):
    save_folder_path = args.save_folder_path
    input_file = args.input
    predictionText = open(save_folder_path + os.path.splitext(input_file)[0] + "_newer.txt", "w+")                       #New level goes here
    predictionSeed = open(save_folder_path + os.path.splitext(input_file)[0] + "_seed.txt", "w+")                       #New level goes here

    #for diversity in [1.0, 1.2]:
    for diversity in [1.0]:
        predictionText.write("Diversity = "+str(diversity)+"\nSeed:\n")
        for _, [X] in enumerate(test_dataloader):
            assert X.shape[0]==1 #batch size should be 1 #Batchify later
            generated = [indices_word[x] for x in X[0].data.numpy()]#since batch size is one in test
            predictionText.write("\n".join(generated)+"\n\n")
            predictionSeed.write("\n".join(generated))
            for _ in range(args.max_output_length):
                preds = get_prediction(model,X,args)[0]##
                next_index = sample(preds.detach().numpy(), diversity)
                next_word = indices_word[next_index]
                generated += [next_word]
                predictionText.write(next_word+"\n")

                next_word = torch.ones(X.shape[0],1)
                X = torch.cat((X,next_word),1)
    predictionText.close()

    return 0

args = parse_args()
dataloader, test_dataloader, (word_indices, indices_word) = create_dataloaders(args, logging.getLogger('get-Data'))

#TODO INIT Weights
print('Build model...')
args.dict_size = len(word_indices)
model = TransformerModel(args)
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8 )
total_steps = len(dataloader)*args.num_epochs
num_warmup_steps = int(total_steps*0.1)# 10 % of the total steps as warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

#TODO Add a saving part too
if os.path.isfile(args.checkpoint_file):
    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

# train the model, output generated text after each iteration
min_loss = 1e10
for epoch in range(1, 2):
    print()
    print('-' * 50)
    print('Iteration', epoch)
    model, optimizer, loss = train_model(model,optimizer,dataloader,args)
    #model.save_weights('GoTweights',overwrite=True)#TODO Save if better
    print("Training loss:",loss)
    #do this when validation loss is reduced
    if loss < min_loss:
        print("Samping outputs now...")
        sample_outputs_from_model(test_dataloader, args)
        model.save(args.save_name,optimizer,args) 
