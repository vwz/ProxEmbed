#encoding=utf-8
'''
proxEmbed model for compute some dataset
'''

import numpy
import theano
from theano import tensor
import lstmModel


def proxEmbedModel(model_options,tparams):
    """
       build ProxEmbed model
    """
    subPaths_matrix=tensor.matrix('subPaths_matrix',dtype='int64')
    subPaths_mask=tensor.matrix('subPaths_mask',dtype=theano.config.floatX)  # @UndefinedVariable
    subPaths_lens=tensor.vector('subPaths_lens',dtype='int64')
    wordsEmbeddings=tensor.matrix('wordsEmbeddings',dtype=theano.config.floatX)  # @UndefinedVariable
    
    def _processSubpath(index):
        length=subPaths_lens[index] 
        x=subPaths_matrix[:length,index:index+1]
        x_mask=subPaths_mask[:length,index:index+1] 
        emb=lstmModel.get_lstm(model_options, tparams, x, x_mask, wordsEmbeddings)
        emb=emb*discountModel(model_options['discount_alpha'], length)
        return emb 
    
    rval,update=theano.scan(
                                _processSubpath,
                                sequences=tensor.arange(subPaths_lens.shape[0]), 
                                )
    emb=0
    if model_options['subpaths_pooling_method']=='mean-pooling': # mean-pooling
        emb = rval.sum(axis=0) 
        emb = emb / rval.shape[0] 
    elif model_options['subpaths_pooling_method']=='max-pooling': # max-pooling
        emb = rval.max(axis=0) 
    else: # default, mean-pooling
        emb = rval.sum(axis=0) 
        emb = emb / rval.shape[0] 
        
    score=tensor.dot(emb,tparams['w'])
    
    return subPaths_matrix,subPaths_mask,subPaths_lens,wordsEmbeddings,score
    
    
def discountModel(alpha,length):
    """
    discount model
    """
    return tensor.exp(alpha*length*(-1))
