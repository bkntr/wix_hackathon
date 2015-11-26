__author__ = 'orent'
import lasagne
import theano
import numpy as np


def learning_rate_func(base_lr, gamma, beta, initial_iter):
    t = theano.shared(np.array(initial_iter, dtype='float32'))
    lr = base_lr * (1.0 + gamma * t) ** beta
    update = (t, t + np.array(1.0, dtype='float32'))
    return lr, update


def nesterov_momentum(loss_or_grads, params, learning_rate=0.01, momentum=0.9, initial_iter=0.0, gamma=0.0001,
                      beta=-0.75):
    """ Similar to lasagne's nesterov momentum updates but uses a decaying learning rate """
    initial_iter = np.array(initial_iter).astype('float32')
    base_lr, momentum, initial_iter = map(lambda x: np.array(x, dtype='float32'),
                                          [learning_rate, momentum, initial_iter])
    lr, lr_update = learning_rate_func(learning_rate, gamma, beta, initial_iter=initial_iter)
    updates = lasagne.updates.nesterov_momentum(loss_or_grads, params, lr, momentum)
    updates[lr_update[0]] = lr_update[1]
    return updates