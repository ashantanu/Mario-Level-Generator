import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
import logging

from torch.utils.data import TensorDataset, DataLoader
input_length = 200
step = 10
column_length = 13
SEPERATOR = " "

def data_one_map_column_words(args, logger):
    path = args.folder_path + args.input
    try: 
        text = open(path).read().lower()
    except UnicodeDecodeError:
        import codecs
        text = codecs.open(path, encoding='utf-8').read().lower()

    logger.info('Single file corpus length:' + str(len(text)))
    logger.info('Parsing each column as a word/single token')

    chars = set(text)
    words = set(open(path).read().lower().split())

    #TODO Check this
    #Adding a start token
    words = list(words)+['<start>']
    words.sort()

    logger.info(f"chars: %s",type(chars))
    logger.info(f"words %s",type(words))
    logger.info(f"total number of unique words: %d",len(words))
    logger.info(f"total number of unique chars: %d", len(chars))

    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    logger.info("word_indices " + str(type(word_indices)) + " length:" + str(len(word_indices)))
    logger.info("indices_words " + str(type(indices_word)) + " length" + str(len(indices_word)))

    maxlen = 30
    step = 3
    logger.info(f"maxlen: %d step: %d", maxlen, step)
    sentences = []
    next_words = []
    next_words= []
    list_words = []

    sentences2=[]
    list_words=text.lower().split()

    for i in range(0,len(list_words)-maxlen, step):
        sentences2 = ' '.join(list_words[i: i + maxlen])
        sentences.append(sentences2)
        next_words.append((list_words[i + maxlen]))
    logger.info(f'nb sequences(length of sentences): %d', len(sentences))
    logger.info(f"length of next_word %d",len(next_words))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen), dtype=np.long)
    y = np.zeros((len(sentences), maxlen+2), dtype=np.long)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence.split()):
            #print(i,t,word)
            X[i, t ] = word_indices[word]
            y[i, t+1 ] = word_indices[word]

        y[i, 0 ] = word_indices['<start>']
        y[i,-1 ] = word_indices[next_words[i]]

    print("Construct Testing Data...")
    start_index = random.randint(0, len(list_words) - maxlen - 1)
    generated = ''
    sentence = list_words[start_index: start_index + maxlen]
    generated += ' '.join(sentence)
    logger.info('----- Generation seed:')
    logger.info( sentence )
    logger.info('------------')
    test = np.zeros((1, maxlen))
    for t, word in enumerate(sentence):
        test[0, t ] = word_indices[word]

    return X, y, test, word_indices, indices_word

#######################################
def join_tiles_in_map(list_chars_columns,method="1"):
    if method=="1":
        return "".join(list_chars_columns), column_length


def data_one_map_tiles(args, logger):
    path = args.folder_path + args.input
    try: 
        text = open(path).read().lower()
    except UnicodeDecodeError:
        import codecs
        text = codecs.open(path, encoding='utf-8').read().lower()

    logger.info('Single file corpus length:' + str(len(text)))
    logger.info('Parsing each column as a word/single token')

    chars = set(text)
    words = set(open(path).read().lower().split())

    #TODO Check this
    #Adding a start token
    chars = list(chars)+['<start>']
    chars.sort()

    logger.info(f"chars: %s",type(chars))
    logger.info(f"words %s",type(words))
    logger.info(f"total number of unique words: %d",len(words))
    logger.info(f"total number of unique chars: %d", len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    logger.info("char_indices " + str(type(char_indices)) + " length:" + str(len(char_indices)))
    logger.info("indices_chars " + str(type(indices_char)) + " length" + str(len(indices_char)))

    maxlen = input_length

    logger.info(f"maxlen: %d step: %d", maxlen, step)
    sentences = []
    next_chars = []
    list_chars = []

    sentences2=[]
    list_chars_columns=text.lower().split()
    list_chars, col_len =join_tiles_in_map(list_chars_columns)

    for i in range(0,len(list_chars)-maxlen, step):
        sentences2 = list_chars[i: i + maxlen]
        sentences.append(sentences2)
        next_chars.append(list_chars[i + maxlen])
    logger.info(f'nb sequences(length of sentences): %d', len(sentences))
    logger.info(f"length of next_word %d",len(next_chars))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen), dtype=np.long)
    y = np.zeros((len(sentences), maxlen+2), dtype=np.long)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t ] = char_indices[char]
            y[i, t+1 ] = char_indices[char]

        y[i, 0 ] = char_indices['<start>']
        y[i,-1 ] = char_indices[next_chars[i]]
    h_x = np.ones((X.shape[0]), dtype=np.long)*col_len

    print("Construct Testing Data...")
    start_index = random.randint(0, len(list_chars) - maxlen - 1)
    sentence = list_chars[start_index: start_index + maxlen]
    logger.info('----- Generation seed:')
    logger.info( [sentence[i:i+col_len] for i in range(0, len(sentence), col_len)] )
    logger.info('------------')
    test = np.zeros((1, maxlen))
    for t, word in enumerate(sentence):
        test[0, t ] = char_indices[word]
    h_x_test = np.ones((test.shape[0]), dtype=np.long)*col_len

    return X, h_x, y, test, h_x_test, char_indices, indices_char


def create_dataloaders(args, logger):
    X, h_x, y, test, h_x_test, word_indices, indices_word = data_one_map_tiles(args, logger)
    print('create dataloader..')
    params = {'batch_size': args.batch_size,
          'shuffle': args.shuffle,
          'num_workers': args.num_workers}
    test_params = {'batch_size': 1,
          'shuffle': args.shuffle,
          'num_workers': args.num_workers}
    dataset = TensorDataset(torch.tensor(X), torch.tensor(h_x), torch.tensor(y))
    test_dataset = TensorDataset(torch.tensor(test), torch.tensor(h_x_test))
    dataloader = DataLoader(dataset, **params)
    val_dataloader = DataLoader(dataset, **params)
    test_dataloader = DataLoader(dataset, **params)
    test_dataloader_sampling = DataLoader(test_dataset, **test_params)

    return dataloader, val_dataloader, test_dataloader, test_dataloader_sampling, (word_indices, indices_word)
 
if __name__ == '__main__':
    class args_class:
        def __init__(self):
            self.input='mario-1-1-edited_trans.txt'
            self.folder_path='../levels_transposed/'
            self.batch_size = 1
            self.shuffle=False
            self.num_workers=1
    args = args_class()
    logging.basicConfig(filename='../logs/test.log',
                            filemode='w',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    dataloader, val_dataloader, test_dataloader, test_dataloader_sampling, (word_indices, indices_word) = create_dataloaders(args, logging.getLogger('Test'))
    print("hello let me test you")