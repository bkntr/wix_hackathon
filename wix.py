__author__ = 'oren'
import re
import os
import sys
import word2vec
import numpy as np
from scipy.sparse import csr_matrix
from nltk.stem.lancaster import LancasterStemmer
import lasagne
from lasagne.layers import InputLayer, DenseLayer, get_output
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from theano import tensor as T


def words_set(words_fname):
    with open(words_fname, 'rt') as fid:
        return set(fid.read().split(','))


w2v_model = word2vec.load('text8.bin')
stop_words = words_set('stop_words.txt')
valids = set([w for w in w2v_model.vocab if w not in stop_words])
stem = LancasterStemmer().stem


def text2bows(text, bows_file='bows.csv'):
    with open(bows_file, 'wt') as bows_fid:
        for idx, t in enumerate(text):
            curr = [pages.strip('()').split('~~')[1].strip('{}').split('~') if '~~' in pages
                    else pages.strip('{}').split('~')
                    for pages in t.strip('{}').split('~~~')]

            curr = set(re.sub('[^a-zA-Z ]', '', ' '.join([cc.strip('()') for c in curr for cc in c])).split(' '))
            curr = [w for w in curr if len(w) > 0 and w in valids]
            bows_fid.write(','.join(curr) + '\n')
            if idx % 1000 == 0:
                sys.stdout.write('\r' + str(idx))
                sys.stdout.flush()


def doc2vecs(words_set, words_dict, bows_file='bows.csv'):
    bows_count = np.memmap('bows_count.bin', dtype='uint16', mode='w+', shape=(len(hashid), len(words_set)))
    bows_binary = np.memmap('bows_binary.bin', dtype='uint8', mode='w+', shape=(len(hashid), len(words_set)))
    with open(bows_file) as docs:
        for idx, line in enumerate(docs):
            for w in line.rstrip('\n').split(','):
                bows_count[idx, words_dict[w]] += 1
                bows_binary[idx, words_dict[w]] = 1
                if idx % 1000 == 0:
                    sys.stdout.write('\r' + str(idx))
                    sys.stdout.flush()


if __name__ == '__main__':
    with open('/home/oren/datahack_sitetext_train_final.csv') as fid:
        wix = fid.read().split('\n')

    hashid, tag, link, text = zip(*[w.split('|') for w in wix if '|' in w])
    unique_tags, tag_idx = np.unique(tag, return_inverse=True)

    # text2bows(text)

    bows = re.split('[,\n]', open('bows.csv').read())
    words_set = set(bows)
    words_dict = dict(zip(words_set, xrange(len(words_set))))
    # doc2vecs(words_set, words_dict)

    bows_count = np.memmap('bows_count.bin', dtype='uint16', mode='r', shape=(len(hashid), len(words_set)))
    bows_binary = np.memmap('bows_binary.bin', dtype='uint8', mode='r', shape=(len(hashid), len(words_set)))

    batch_size = 32
    hidden_units = [256, 128]
    sig_width = 100

    input_var = T.fmatrix('inputs')
    target_var = T.ivector('targets')

    input_layer = InputLayer(shape=(32, len(bows_binary.shape[1])), name='input_layer', input_var=input_var)
    hidden = DenseLayer(input_layer, hidden_units[0])
    for ne in hidden_units[1:]:
        hidden = DenseLayer(hidden, ne)
    output_layer = DenseLayer(hidden, sig_width, nonlinearity=softmax)
    prediction = get_output(output_layer)

    loss = categorical_crossentropy(prediction, target_var)









    

