#encoding=utf-8

from __future__ import print_function
import six.moves.cPickle as pickle  # @UnresolvedImport

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config 
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import toolsFunction

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)  # @UndefinedVariable


def _p(pp, name):
    return '%s_%s' % (pp, name)


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    """
    generate lstm
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1] 
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim] 

    def _step(m_, x_, h_, c_): 
        preact = tensor.dot(h_, tparams['lstm_U']) 
        preact += x_ 

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dimension'])) # input gate 
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dimension'])) # forget gate 
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dimension'])) # output gate 
        c = tensor.tanh(_slice(preact, 3, options['dimension'])) #  cell 

        c = f * c_ + i * c 
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c) 
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    state_below = (tensor.dot(state_below, tparams['lstm_W']) + tparams['lstm_b'])

    dim_proj = options['dimension']
    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), 
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.), 
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps) # maxlen
    return rval[0] 



def build_model(tparams, options, x, mask, wordsemb):
    """
    build the model
    """
    n_timesteps = x.shape[0] 
    n_samples = x.shape[1] 
    emb = wordsemb[x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['word_dimension']])
    proj = lstm_layer(tparams, emb, options,
                                            prefix='lstm',
                                            mask=mask)
    output=None
    if options['h_output_method'] == 'h': # the last h as the output
        temp=proj[-1] 
        output=temp[0] 
    elif options['h_output_method'] == 'mean-pooling': # mean-pooling as the output
        temp1 = (proj * mask[:, :, None]).sum(axis=0) 
        temp2 = temp1 / mask.sum(axis=0)[:, None]
        output=temp2[0]
    elif options['h_output_method'] == 'max-pooling': # max-pooling as the output
        temp1=proj * mask[:, :, None] 
        temp2=temp1.sum(axis=1) 
        output = temp2.max(axis=0) 
    else : # default, the last h as the output
        temp=proj[-1]
        output=temp[0] 
    return  output


# get lstm model by parameters
def get_lstm(
    model_options, # the options parameters for the model
    tparams, # theano shared variables
    x, # a sub-path
    x_mask, # the mask of this sub-path
    wordsemb, # embeddings of all words
):

    # build the model
    proj = build_model(tparams, model_options, x, x_mask, wordsemb)
    return proj
