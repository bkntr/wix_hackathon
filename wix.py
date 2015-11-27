__author__ = 'oren'
import re
import os
import sys
import word2vec
import numpy as np
from nltk.stem.porter import PorterStemmer
import theano
import lasagne
from nesterov_momentum import nesterov_momentum
from lasagne.layers import InputLayer, DenseLayer, get_output, DropoutLayer
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.regularization import regularize_network_params, l2, l1
from theano import tensor as T


def words_set(words_fname):
    with open(words_fname, 'rt') as fid:
        return set(fid.read().split(','))


w2v_model = word2vec.load('text8.bin')
stop_words = words_set('stop_words.txt')
valids = set([w for w in w2v_model.vocab if w not in stop_words])
stem = PorterStemmer().stem


def text2bows(text, bows_file='bows.csv'):
    with open(bows_file, 'wt') as bows_fid:
        for idx, t in enumerate(text):
            curr = [pages.strip('()').split('~~')[1].strip('{}').split('~') if '~~' in pages
                    else pages.strip('{}').split('~')
                    for pages in t.strip('{}').split('~~~')]

            curr = set(re.sub('[^a-zA-Z ]', '', ' '.join([cc.strip('()') for c in curr for cc in c])).split(' '))
            curr = [stem(w) for w in curr if len(w) > 0 and w in valids]
            bows_fid.write(','.join(curr) + '\n')
            if idx % 1000 == 0:
                sys.stdout.write('\r' + str(idx))
                sys.stdout.flush()


def doc2vecs(words_set, words_dict, bows_file='bows.csv', bc='bows_count.bin', bb='bows_binary.bin'):
    bows_count = np.memmap(bc, dtype='uint16', mode='w+', shape=(len(hashid), len(words_set)))
    bows_binary = np.memmap(bb, dtype='uint8', mode='w+', shape=(len(hashid), len(words_set)))
    with open(bows_file) as docs:
        for idx, line in enumerate(docs):
            for w in line.rstrip('\n').split(','):
                if w in words_dict:
                    bows_count[idx, words_dict[w]] += 1
                    bows_binary[idx, words_dict[w]] = 1
                    if idx % 1000 == 0:
                        sys.stdout.write('\r' + str(idx))
                        sys.stdout.flush()


if __name__ == '__main__':
    with open('../notebooks/datahack_sitetext_train_final.csv') as fid:
        wix = fid.read().split('\n')

    hashid, tag, link, _ = zip(*[w.split('|') for w in wix if '|' in w])
    unique_tags, tag_idx = np.unique(tag, return_inverse=True)

    # text2bows(text)

    bows = re.split('[,\n]', open('bows_stem.csv').read())
    words_set = set(bows)
    words_dict = dict(zip(words_set, xrange(len(words_set))))
    # doc2vecs(words_set, words_dict)

    bows_count = np.memmap('bows_count_stem.bin', dtype='uint16', mode='r', shape=(len(hashid), len(words_set)))
    bows_binary = np.memmap('bows_binary_stem.bin', dtype='uint8', mode='r', shape=(len(hashid), len(words_set)))

    idf = np.log(np.float32(len(bows_binary)) / (np.sum(bows_binary, axis=0)))

    """ split train/test """

    train_test_ratio = 1.0
    N = len(tag_idx)
    train_len = int(train_test_ratio * N)
    shuffle_idx = np.random.permutation(N)
    train_idx = shuffle_idx[:train_len]
    test_idx = shuffle_idx[train_len:]

    """ train model """

    input_data = bows_count
    batch_size = 32
    hidden_units = [256, 128]

    input_var = T.fmatrix('inputs')
    target_var = T.ivector('targets')

    input_layer = InputLayer(shape=(batch_size, input_data.shape[1]), name='input_layer', input_var=input_var)
    hidden = [DenseLayer(input_layer, hidden_units[0])]
    for ne in hidden_units[1:]:
        hidden.append(DenseLayer(hidden[-1], ne))
    output_layer = DenseLayer(hidden[-1], len(unique_tags), nonlinearity=softmax)
    prediction = get_output(output_layer)

    loss = categorical_crossentropy(prediction, target_var)  # + 0.0001 * regularize_network_params(hidden[0], l1)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    # updates = adam(loss, params, learning_rate=0.001)
    train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)
    test_fn = theano.function([input_var, target_var], [loss, prediction])

    print_interval = 100
    test_interval = 1000
    test_size = 100
    iter_idx = 0
    epoch_idx = 0
    stats_accum_train = dict(loss=0.0, acc=0.0, count=0.0)
    while True:
        print '\033[94m' + 'epoch #{}'.format(epoch_idx) + '\033[0m'
        epoch_idx += 1
        np.random.shuffle(train_idx)
        for train_iter in xrange(0, len(train_idx) + batch_size, batch_size):
            curr_idx = train_idx[train_iter:min(train_iter + batch_size, len(train_idx))]
            # curr_bows_train = (np.log(1.0 + input_data[curr_idx]) * idf).astype('float32')
            curr_bows_train = (input_data[curr_idx] * idf).astype('float32')
            if len(curr_bows_train) != batch_size:
                break
            curr_tags_train = tag_idx[curr_idx].astype('int32')
            curr_loss, curr_pred = train_fn(curr_bows_train, curr_tags_train)
            curr_pred = np.argmax(curr_pred, axis=1)
            curr_acc = np.sum(curr_pred == curr_tags_train).astype('float32') / len(curr_tags_train)
            stats_accum_train['loss'] += curr_loss
            stats_accum_train['acc'] += curr_acc
            stats_accum_train['count'] += 1
            iter_idx += 1

            if iter_idx % print_interval == 0:
                print '{})  loss: {}  acc: {}'.format(iter_idx,
                                                      stats_accum_train['loss'] / stats_accum_train['count'],
                                                      100.0 * stats_accum_train['acc'] / stats_accum_train['count'])
                stats_accum_train = dict(loss=0.0, acc=0.0, count=0.0)

            if False:
                if iter_idx % test_interval == 0:
                    stats_accum_test = dict(loss=0.0, acc=0.0, count=0.0)
                    for test_iter in xrange(test_size):
                        curr_idx = test_idx[np.random.choice(len(test_idx), batch_size)]
                        # curr_bows_test = (np.log(1.0 + input_data[curr_idx]) * idf).astype('float32')
                        curr_bows_test = (input_data[curr_idx] * idf).astype('float32')
                        curr_tags_test = tag_idx[curr_idx].astype('int32')
                        curr_loss, curr_pred = test_fn(curr_bows_test, curr_tags_test)
                        curr_pred = np.argmax(curr_pred, axis=1)
                        curr_acc = np.sum(curr_pred == curr_tags_test).astype('float32') / len(curr_tags_test)
                        stats_accum_test['loss'] += curr_loss
                        stats_accum_test['acc'] += curr_acc
                        stats_accum_test['count'] += 1
                    res_text = '{})  loss: {}  acc: {}'.format(iter_idx,
                                                               stats_accum_test['loss'] / stats_accum_test['count'],
                                                               100.0 * stats_accum_test['acc'] / stats_accum_test['count'])
                    print '\033[92m' + res_text + '\033[0m'
        break

    """ review model """

    error_indices = []
    error_predictions = []
    for test_iter in xrange(0, len(test_idx) + batch_size, batch_size):
        curr_idx = slice(test_iter, min(test_iter + batch_size, len(test_idx)))
        curr_bows_test = (np.log(1.0 + input_data[test_idx[curr_idx]]) * idf).astype('float32')
        if len(curr_bows_test) == 0:
            break
        curr_tags_test = tag_idx[test_idx[curr_idx]].astype('int32')
        curr_loss, curr_pred = test_fn(curr_bows_test, curr_tags_test)
        curr_pred = np.argmax(curr_pred, axis=1)
        local_errors = np.where(curr_pred != curr_tags_test)[0]
        global_errors = local_errors + test_iter
        error_indices.extend(list(test_idx[global_errors]))
        error_predictions.extend([unique_tags[p] for p in curr_pred[local_errors]])

    """ evaluate """

    with open('datahack_sitetext_test_final.csv') as fid:
        wix = fid.read().split('\n')

    hashid, tag, link, text = zip(*[w.split('|') for w in wix if '|' in w])

    text2bows(text, bows_file='bows_eval.csv')

    bows = re.split('[,\n]', open('bows_stem.csv').read())
    words_set = set(bows)
    words_dict = dict(zip(words_set, xrange(len(words_set))))
    doc2vecs(words_set, words_dict, 'bows_eval.csv', bc='bows_count_stem_eval.bin', bb='bows_binary_stem_eval.bin')

    bows_count = np.memmap('bows_count_stem_eval.bin', dtype='uint16', mode='r', shape=(len(hashid), len(words_set)))
    bows_binary = np.memmap('bows_binary_stem_eval.bin', dtype='uint8', mode='r', shape=(len(hashid), len(words_set)))

    den = np.sum(bows_binary, axis=0)
    den[den == 0] = 1.0
    idf = np.log(np.float32(len(bows_binary)) / den)

    predictions = []
    for test_iter in xrange(0, len(bows_count) + batch_size, batch_size):
        curr_idx = slice(test_iter, min(test_iter + batch_size, len(bows_count)))
        # curr_bows_test = (np.log(1.0 + input_data[test_idx[curr_idx]]) * idf).astype('float32')
        curr_bows_test = (bows_count[curr_idx] * idf).astype('float32')
        if len(curr_bows_test) == 0:
            break
        if len(curr_bows_test) != batch_size:
            curr_bows_test = np.vstack([curr_bows_test, np.tile(curr_bows_test[-1:], [batch_size - len(curr_bows_test), 1])])
        curr_loss, curr_pred = test_fn(curr_bows_test, curr_tags_test)
        curr_pred = unique_tags[np.argmax(curr_pred, axis=1)]
        predictions.extend(curr_pred)
    predictions = predictions[:len(bows_binary)]
    with open('results_linked.csv', 'wt') as fid:
        fid.write('\n'.join(['|'.join(z) for z in zip(hashid, predictions, link)]))
    with open('results.csv', 'wt') as fid:
        fid.write('\n'.join(['|'.join(z) for z in zip(hashid, predictions)]))





    

