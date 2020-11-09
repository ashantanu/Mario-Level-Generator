import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
import logging
import pickle

from torch.utils.data import TensorDataset, DataLoader
column_length = 13
SEPERATOR = " "
input_length = 10*13#TODO change this for another version of dataset independent of columns
step = 13#TODO change this for another version of dataset independent of columns
output_length = 2*13#TODO change this for another version of dataset independent of columns
start_token="</s>"

#######################################
def join_columns_in_map(list_chars_columns,method="1"):
    if method=="1":
        return "".join(list_chars_columns), column_length

def data_from_text_files(args, logger):
    if os.path.exists(args.dataset_path):
        with open(args.dataset_path,"rb") as f:
            return pickle.load(f)

    path = args.folder_path
    games = [x for x in os.listdir(path) if x in args.games_to_use]
    all_text = []

    #load all the text
    for game in games:
        levels = os.listdir(os.path.join(path,game))
        levels = [x for x in levels if '.txt' in x]
        for level in levels:
            path_ = os.path.join(path,game,level)
            try: 
                text = open(path_).read().lower()
            except UnicodeDecodeError:
                import codecs
                text = codecs.open(path_, encoding='utf-8').read().lower()
            all_text.append(text)

    logger.info('Number of levels:' + str(len(all_text)))
    logger.info('Average file corpus length:' + str(np.mean([len(x) for x in all_text])))
    logger.info('Parsing each column as a word/single token')

    chars = set("".join(all_text))
    words = "\n".join(all_text).split()

    #Adding a start token
    chars = list(chars)+[start_token]
    chars.sort()

    logger.info(f"total number of unique words: %d",len(words))
    logger.info(f"total number of unique chars: %d", len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    logger.info("char_indices " + str(type(char_indices)) + " length:" + str(len(char_indices)))
    logger.info("indices_chars " + str(type(indices_char)) + " length" + str(len(indices_char)))


    logger.info(f"maxlen: %d step: %d", input_length, step)

    #split levels into train, test, val
    train_levels = random.sample(all_text,int(len(all_text)*0.7))
    val_levels = random.sample([x for x in all_text if x not in train_levels],int(len(all_text)*0.15))
    test_levels = random.sample([x for x in all_text if x not in train_levels and x not in val_levels],int(len(all_text)*0.15))

    logger.info("Constructing datasets")
    train = get_dataset_using_char_map(train_levels, logger, char_indices)
    val = get_dataset_using_char_map(val_levels, logger, char_indices)
    test = get_dataset_using_char_map(test_levels, logger, char_indices)
    logger.info(f"dataset shapes:\nTrain:%s\nVal:%s\nTest:%s\n",str(train[0].shape),str(val[0].shape),str(test[0].shape))
    return train, val, test, char_indices, indices_char

def get_dataset_using_char_map(all_text, logger, char_indices):
    sentences = []
    next_chars = []
    list_chars = []
    maxlen = input_length
    sentences2=[]
    for text in all_text:
        list_chars_columns=text.lower().split()
        list_chars, col_len =join_columns_in_map(list_chars_columns)

        for i in range(0,len(list_chars)-maxlen-output_length, step):
            sentences2 = list_chars[i: i + maxlen]
            sentences.append(sentences2)
            next_chars.append(list_chars[i + maxlen:i + maxlen + output_length])
    logger.info(f'number of sequences(length of sentences): %d', len(sentences))
    logger.info(f"length of next_word %d",len(next_chars))

    X = np.zeros((len(sentences), maxlen), dtype=np.long)
    y = np.zeros((len(sentences), output_length), dtype=np.long)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t ] = char_indices[char]
        for t, char in enumerate(next_chars[i]):
            y[i, t ] = char_indices[char]

        #y[i, 0 ] = char_indices['<start>']
    h_x = np.ones((X.shape[0]), dtype=np.long)*col_len
    return X, h_x, y

def create_dataloader(args, logger, dataset, sampling=False):
    X, h_x, y = dataset
    print('create dataloader..')
    batch = 1 if sampling else args.batch_size
    params = {'batch_size': batch,
          'shuffle': args.shuffle,
          'num_workers': args.num_workers}
    dataset = TensorDataset(torch.tensor(X), torch.tensor(h_x), torch.tensor(y))
    dataloader = DataLoader(dataset, **params)
    return dataloader
 
if __name__ == '__main__':
    class args_class:
        def __init__(self):
            self.input='mario-1-1-edited_trans.txt'
            self.folder_path='../levels_transposed/'
            self.batch_size = 1
            self.shuffle=False
            self.num_workers=1
            self.games_to_use=['Original']
            self.dataset_path = "../cached_data/dataset.pkl"
    args = args_class()
    logging.basicConfig(filename='../logs/test.log',
                            filemode='w',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    #dataloader, val_dataloader, test_dataloader, test_dataloader_sampling, (word_indices, indices_word) = create_dataloaders(args, logging.getLogger('Test'))
    logger = logging.getLogger('Test')
    train, val, test, char_indices, indices_char = data_from_text_files(args, logger)
    dataloaders = []
    for data in [train,val,test]:
        dataloaders.append(create_dataloader(args, logger, data, sampling=False))
    dataloaders.append(create_dataloader(args, logger, test, sampling=True))
    print("hello let me test you")