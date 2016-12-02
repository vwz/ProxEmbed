#encoding=utf-8
'''
Generate ProxEmbed Model
'''
import numpy
import theano
from theano import tensor
import lstmModel
from theano.ifelse import ifelse


def proxEmbedModel(model_options,tparams):
    """
    generate proxEmbed model
    """
    trainingParis=tensor.tensor3('trainingParis',dtype='int64') 
    subPaths_matrix=tensor.matrix('subPaths_matrix',dtype='int64') 
    subPaths_mask=tensor.matrix('subPaths_mask',dtype=theano.config.floatX)  # @UndefinedVariable 
    subPaths_lens=tensor.vector('subPaths_lens',dtype='int64') 
    wordsEmbeddings=tensor.matrix('wordsEmbeddings',dtype=theano.config.floatX)  # @UndefinedVariable 
    
    def _processTriple(fourPairs,lossSum):
        
        def _processSubpath(index):
            length=subPaths_lens[index] 
            x=subPaths_matrix[:length,index:index+1]
            x_mask=subPaths_mask[:length,index:index+1] 
            emb=lstmModel.get_lstm(model_options, tparams, x, x_mask, wordsEmbeddings)
            emb=emb*discountModel(model_options['discount_alpha'], length)
            return emb 
        
        def iftFunc():
            embx=numpy.zeros(model_options['dimension'],) 
            embx.astype(theano.config.floatX)  # @UndefinedVariable
            return embx
         
        def iffFunc(start,end):
            embx=None
            rval,update=theano.scan(
                                _processSubpath,
                                sequences=tensor.arange(start,end), 
                                )
            if model_options['subpaths_pooling_method']=='mean-pooling': # mean-pooling
                embx = rval.sum(axis=0) 
                embx = embx / rval.shape[0] 
            elif model_options['subpaths_pooling_method']=='max-pooling': # max-pooling
                embx = rval.max(axis=0) 
            else: # default, mean-pooling
                embx = rval.sum(axis=0)
                embx = embx / rval.shape[0] 
                
            return embx
        
        start=fourPairs[0][0] 
        end=fourPairs[0][1] 
        emb1=None 
        emb1=ifelse(tensor.eq(start,end),iftFunc(),iffFunc(start,end)) # 先选一个，然后计算这个值
        
        start=fourPairs[2][0] 
        end=fourPairs[2][1]
        emb2=None 
        emb2=ifelse(tensor.eq(start,end),iftFunc(),iffFunc(start,end)) # 先选一个，然后计算这个值
            
        loss=0
        param=model_options['objective_function_param'] 
        if model_options['objective_function_method']=='sigmoid': 
            loss=-tensor.log(tensor.nnet.sigmoid(param*(tensor.dot(emb1,tparams['w'])-tensor.dot(emb2,tparams['w'])))) # sigmoid
        
        return loss+lossSum
        
    rval,update=theano.scan(
                            _processTriple,
                            sequences=trainingParis, 
                            outputs_info=tensor.constant(0., dtype=theano.config.floatX), # @UndefinedVariable # 输出是这个triple的loss
                            )
    cost=rval[-1]
    cost+=model_options['decay_lstm_W']*(tparams['lstm_W'] ** 2).sum()
    cost+=model_options['decay_lstm_U']*(tparams['lstm_U'] ** 2).sum()
    cost+=model_options['decay_lstm_b']*(tparams['lstm_b'] ** 2).sum()
    cost+=model_options['decay_w']*(tparams['w'] ** 2).sum()
    return trainingParis, subPaths_matrix, subPaths_mask, subPaths_lens , wordsEmbeddings, cost


def discountModel(alpha,length):
    """
    discount model
    """
    return tensor.exp(alpha*length*(-1))
    
def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)  # @UndefinedVariable