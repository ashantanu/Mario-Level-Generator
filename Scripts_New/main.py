import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
import argparse
import logging
import io

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#from models import TransformerModel
from GPT2_Model import GPT2Config, GPT2LMHeadModel
from create_input import data_from_text_files, create_dataloader, column_length
from init_params import *

logging.basicConfig(filename='../logs/model.log',
                            filemode='w',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

def get_hyperparameter_string(args):
    hyperparamters = [args.suffix, args.emb_size, args.max_sample_input]
    for i, param in enumerate(hyperparamters):
        '''
        if param == args.dataloader:
            param = args.dataloader.split("/")
            param = param[-2] if param[-1] == "" else param[-1]
            param = param.split(".pth")[0]
            hyperparamters[i] = param'''

    hyperparameter_string = "_".join([str(x) for x in hyperparamters])
    return hyperparameter_string

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default = "../levels_transposed/", help = 'folder_path')
    parser.add_argument('--save_folder_path', type=str, default='../levels_prediction_textfiles/',help='Output data file path')
    parser.add_argument('--dataset_path', type=str, default='../cached_data/dataset.pkl',help='data path')
    parser.add_argument('--log_path', type=str, default='../model_logs/',help='metrics, model checkpoints etc data file path')
    parser.add_argument('--input', type=str, default='mario-1-1-edited_trans.txt',help='Input data file path')
    parser.add_argument('--checkpoint_file', type=str, default='model_checkpoint.pt',help='File path for saving model')
    parser.add_argument('--suffix', type=str, default='V1',help='suffix for saving model')

    #model args 
    parser.add_argument('--emb_size', type=int, default=256,help='Embedding size')
    parser.add_argument('--n_layers', type=int, default=1,help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8,help='Number of layers')

    #traning args
    parser.add_argument('--batch_size', type=int, default=8,help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,help='Number of workers')
    parser.add_argument('--num_epochs', type=int, default=2,help='Number of Epochs')
    parser.add_argument('--shuffle', type=bool, default=False,help='Shuffle dataset')
    parser.add_argument('--device', type=str, default='cpu',help='cpu or gpu/cuda')
    
    #sampling args
    parser.add_argument('--max_output_length', type=int, default=26,help='max output length')
    parser.add_argument('--max_sample_input', type=int, default=30,help='max length of input for generating sample output')
    parser.add_argument('--temperature', type=int, default=1.0,help='temperature for sampling')
    parser.add_argument("--no_sample", type=bool, default=True, help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    
    args = parser.parse_args()
    args.games_to_use = games_to_use
    hyperparameter_string = get_hyperparameter_string(args)
    args.save_path = args.log_path + "BestCheckpoint_"+ hyperparameter_string + ".pth"
    args.metric_save_path = args.log_path + "BestMetrics_"+ hyperparameter_string + ".txt"
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    return args

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def train_model(model,optimizer,dataloader,args):
    loss_total = 0
    #TODO Move batching to data loader
    for iteration, (X, h_x, y) in enumerate(dataloader):
        for p in model.parameters(): p.grad = None
        X = X.to(args.device)
        h_x = h_x.to(args.device)
        y = y.to(args.device)
        model.train()
        loss = model.get_loss(X, h_x, y)
        loss.backward()
        optimizer.step()
        loss_total+=loss.detach().data

        if iteration%100==0:
            print("Iteration step:",iteration)

    return model, optimizer, loss_total

def eval_model(model,optimizer,dataloader,args):
    loss_total = 0
    #TODO Move batching to data loader
    for _, (X, h_x, y) in enumerate(dataloader):
        for p in model.parameters(): p.grad = None
        X = X.to(args.device)
        h_x = h_x.to(args.device)
        y = y.to(args.device)
        model.eval()
        with torch.no_grad():
            loss = model.get_loss(X, h_x, y)
            loss_total+=loss.detach().data

    return loss_total

def train_gpt(model,optimizer,dataloader,args):
    loss_total = 0
    #TODO Move batching to data loader
    for iteration, (X, h_x, y) in enumerate(dataloader):
        X = X.to(args.device)
        h_x = h_x.to(args.device)
        y = y.to(args.device)
        input_ = torch.cat((X,y),axis=1)
        labels = torch.cat((torch.ones(X.shape)*-1,y),axis=1).to(torch.long).to(args.device)
        model.train()
        for p in model.parameters(): p.grad = None
        loss = model(input_,lm_labels=labels)
        loss.backward()
        optimizer.step()
        loss_total+=loss.detach().data

        if iteration%100==0:
            print("Iteration step:",iteration)

    return model, optimizer, loss_total

def eval_gpt(model,optimizer,dataloader,args):
    loss_total = 0
    #TODO Move batching to data loader
    for _, (X, h_x, y) in enumerate(dataloader):
        for p in model.parameters(): p.grad = None
        X = X.to(args.device)
        h_x = h_x.to(args.device)
        y = y.to(args.device)
        input_ = torch.cat((X,y),axis=1)
        labels = torch.cat((torch.ones(X.shape)*-1,y),axis=1).to(torch.long).to(args.device)
        model.eval()
        with torch.no_grad():
            loss = model(input_,lm_labels=labels)
            loss_total+=loss.detach().data

    return loss_total

def get_prediction_gpt(model, X, h_x, args):
    X = X[:,-(args.n_positions-1):]
    model.eval()
    X = X.to(args.device).to(torch.long)
    h_x = h_x.to(args.device).to(torch.long)
    with torch.no_grad():
        logits, _ = model(X)

    if isinstance(logits, tuple):  # for gpt2 and maybe others
        logits = logits[0]
    logits = logits[0, -1, :] / args.temperature
    logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
    probs = F.softmax(logits, dim=-1)
    return probs

def get_prediction(model, X, h_x, args):
    X = X[:,-args.max_sample_input:]
    model.eval()
    X = X.to(args.device).to(torch.long)
    h_x = h_x.to(args.device).to(torch.long)
    with torch.no_grad():
        pred = model.predict(X, h_x)
    return pred[:,-1,:].detach()

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def sample_sequence_gpt(test_dataloader, model, args, indices_char):
    save_folder_path = args.save_folder_path
    predictionText = open(save_folder_path + "sampled_output.txt", "w+")#New level goes here

    predictionText.write("{[\n")
    for i, (X, h_x, _) in enumerate(test_dataloader):
        assert X.shape[0]==1 #batch size should be 1 #Batchify later
        seed = [indices_char[x] for x in X[0].cpu().numpy()]#since batch size is one in test
        if i>0:
            predictionText.write(",\n")
        predictionText.write("{'seed':"+"".join(seed)+"\n,\n")
        current_output = []
        for _ in range(args.max_output_length):
            input_ = torch.tensor(torch.cat((X,torch.tensor(current_output).unsqueeze(0)),axis=1), device=args.device)
            probs = get_prediction_gpt(model, input_, h_x, args)
            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
            current_output.append(prev.item())
        
        generated = [indices_char[x] for x in current_output]
        predictionText.write("'generated':"+"".join(generated)+"\n}]")
    
    predictionText.write("}")
    predictionText.close()

def sample_outputs_from_model(test_dataloader,args):
    save_folder_path = args.save_folder_path
    input_file = args.input
    predictionText = open(save_folder_path + os.path.splitext(input_file)[0] + "_newer.txt", "w+")                       #New level goes here
    predictionSeed = open(save_folder_path + os.path.splitext(input_file)[0] + "_seed.txt", "w+")                       #New level goes here

    #for diversity in [1.0, 1.2]:
    for diversity in [1.0]:
        predictionText.write("Diversity = "+str(diversity)+"\nSeed:\n")
        for _, (X, h_x) in enumerate(test_dataloader):
            assert X.shape[0]==1 #batch size should be 1 #Batchify later
            generated = [indices_char[x] for x in X[0].data.numpy()]#since batch size is one in test
            predictionText.write("\n".join(generated)+"\n\n")
            predictionSeed.write("\n".join(generated))
            for _ in range(args.max_output_length):
                preds = get_prediction(model, X, h_x, args)[0]##
                next_index = sample(preds.detach().numpy(), diversity)
                next_word = indices_char[next_index]
                generated += [next_word]
                predictionText.write(next_word+"\n")

                next_word = torch.ones(X.shape[0],1)
                X = torch.cat((X,next_word),1)
    predictionText.close()

    return 0

def run_model():
    args = parse_args()
    writer = SummaryWriter(args.log_path,filename_suffix='MarioGPT',comment=get_hyperparameter_string(args))

    data_logger = logging.getLogger('Get-Data')
    train, val, test, char_indices, indices_char = data_from_text_files(args, data_logger)
    train, val, test = [x[:5] for x in train], [x[:5] for x in train], [x[:5] for x in train]#when testing
    dataloader = create_dataloader(args, data_logger, train, sampling=False)
    val_dataloader = create_dataloader(args, data_logger, val, sampling=False)
    test_dataloader = create_dataloader(args, data_logger, test, sampling=False)
    test_dataloader_sampling = create_dataloader(args, data_logger, test, sampling=True)

    #TODO INIT Weights
    print('Build model...')
    args.dict_size = len(char_indices)
    args.column_length = column_length
    args.n_positions = 256
    args.n_ctx = 256
    gpt2config = GPT2Config(args)
    model = GPT2LMHeadModel(gpt2config)
    #model = TransformerModel(args)
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8 )
    total_steps = len(dataloader)*args.num_epochs
    num_warmup_steps = int(total_steps*0.1)# 10 % of the total steps as warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    if os.path.isfile(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Resuming from: Epoch:",epoch, " Loss:", loss)

    # train the model, output generated text after each iteration
    min_loss = 1e10
    for epoch in range(args.num_epochs):
        print()
        print('-' * 50)
        print('Iteration', epoch)
        model, optimizer, loss = train_gpt(model,optimizer,dataloader,args)
        print("Training loss:",loss.data)

        val_loss = eval_gpt(model,optimizer,val_dataloader,args)
        print("Validation loss:",val_loss.data)

        test_loss = eval_gpt(model,optimizer,test_dataloader,args)
        print("Test loss:",val_loss.data)

        writer.add_scalar('Loss/train',loss.data,epoch)
        writer.add_scalar('Loss/val',val_loss.data,epoch)
        writer.add_scalar('Loss/test',test_loss.data,epoch)
        if val_loss < min_loss:
            print("Performance improved...")
            model.save(args.save_path,optimizer,args,epoch,loss) 
            #sample_outputs_from_model(test_dataloader_sampling, args)
            sample_sequence_gpt(test_dataloader_sampling, model, args, indices_char)
            with io.open(args.metric_save_path, 'w', encoding='utf8') as f:
                f.write("Val Loss:{:.5f}, Test Loss:{:.5f}".format(val_loss.data,test_loss.data))

if __name__ == "__main__":
    run_model()