__author__ = 'oren'
import re
import os
import sys
import word2vec
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
import theano
import lasagne
from nesterov_momentum import nesterov_momentum
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
    with open('../notebooks/datahack_sitetext_train_final.csv') as fid:
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

    """ split train/test """

    train_test_ratio = 0.9
    N = len(tag_idx)
    train_len = train_test_ratio * N
    shuffle_idx = np.random.permutation(N)
    shuffled_tags = tag_idx[shuffle_idx]
    shuffled_bows = bows_binary[shuffle_idx]
    tags_train = shuffled_tags[:train_len]
    tags_test = shuffled_tags[train_len:]
    bows_train = shuffled_bows[:train_len]
    bows_test = shuffled_bows[train_len:]

    """ train model """

    batch_size = 32
    hidden_units = [256, 128]
    sig_width = 100

    input_var = T.fmatrix('inputs')
    target_var = T.ivector('targets')

    input_layer = InputLayer(shape=(32, bows_binary.shape[1]), name='input_layer', input_var=input_var)
    hidden = DenseLayer(input_layer, hidden_units[0])
    for ne in hidden_units[1:]:
        hidden = DenseLayer(hidden, ne)
    output_layer = DenseLayer(hidden, sig_width, nonlinearity=softmax)
    prediction = get_output(output_layer)

    loss = categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)
    test_fn = theano.function([input_var, target_var], [loss, prediction])

    print_interval = 100
    test_interval = 100
    test_size = 10
    iter_idx = 0
    stats_accum_train = dict(loss=0.0, acc=0.0, count=0.0)
    while True:
        iter_idx += 1
        curr_idx = np.random.choice(len(bows_train), batch_size)
        curr_bows_train = bows_train[curr_idx]
        curr_tags_train = tags_train[curr_idx].astype('int32')
        curr_loss, curr_pred = train_fn(curr_bows_train, curr_tags_train)
        curr_pred = np.argmax(curr_pred, axis=1)
        curr_acc = np.sum(curr_pred == curr_tags_train).astype('float32') / len(curr_tags_train)
        stats_accum_train['loss'] += curr_loss
        stats_accum_train['acc'] += curr_acc
        stats_accum_train['count'] += 1

        if iter_idx % test_interval == 0:
            stats_accum_test = dict(loss=0.0, acc=0.0, count=0.0)
            for test_iter in xrange(test_size):
                curr_idx = np.random.choice(len(bows_test), batch_size)
                curr_bows_test = bows_test[curr_idx]
                curr_tags_test = tags_test[curr_idx].astype('int32')
                curr_loss, curr_pred = test_fn(curr_bows_test, curr_tags_test)
                curr_pred = np.argmax(curr_pred, axis=1)
                curr_acc = np.sum(curr_pred == curr_tags_test).astype('float32') / len(curr_tags_test)
                stats_accum_test['loss'] += curr_loss
                stats_accum_test['acc'] += curr_acc
                stats_accum_test['count'] += 1
                res_text = '{})  loss: {}  acc: {}'.format(iter_idx, stats_accum['loss'] / stats_accum['count'],
                                                           100.0 * stats_accum['acc'] / stats_accum['count'])
                print '\033[92m' + res_text + '\033[0m'

        if iter_idx % print_interval == 0:
            print '{})  loss: {}  acc: {}'.format(iter_idx,
                                                  stats_accum['loss'] / stats_accum['count'],
                                                  100.0 * stats_accum['acc'] / stats_accum['count'])
            stats_accum = dict(loss=0.0, acc=0.0, count=0.0)








    

