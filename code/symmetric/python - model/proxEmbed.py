#encoding=utf-8

import dataProcessTools
import numpy
import theano
from theano import tensor
from theano import config
from collections import OrderedDict
import time
import six.moves.cPickle as pickle  # @UnresolvedImport
import proxEmbedModelMulti


# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)  # @UndefinedVariable

def adadelta(lr, tparams, grads, fourPairs, subPaths_matrix, subPaths_mask, subPaths_lens, wemb, cost):
    """
    An adaptive learning rate optimizer adadelta
        
    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([fourPairs, subPaths_matrix, subPaths_mask, subPaths_lens, wemb], cost, updates=zgup + rg2up,
                                    on_unused_input='ignore',
                                    name='adadelta_f_grad_shared')
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) 
             for ru2, ud in zip(running_up2, updir)] 
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)] 
    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

        
def ortho_weight(ndim):
    """
        initialize a matrix 
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)  # @UndefinedVariable

def init_params_weight(row,column):
    """
    initialize matrix parameters by row and column
    """
    lstm_W = numpy.random.rand(row, column) 
    return lstm_W.astype(config.floatX)  # @UndefinedVariable


def init_sharedVariables(options):
    """
        initialize all the shared parameters
    """
    print 'init shared Variables......'
    params = OrderedDict()
    lstm_W=numpy.concatenate([
                              init_params_weight(options['word_dimension'],options['dimension']),
                              init_params_weight(options['word_dimension'],options['dimension']),
                              init_params_weight(options['word_dimension'],options['dimension']),
                              init_params_weight(options['word_dimension'],options['dimension'])
                              ],axis=1)
    params['lstm_W'] = lstm_W
    lstm_U = numpy.concatenate([ortho_weight(options['dimension']),
                           ortho_weight(options['dimension']),
                           ortho_weight(options['dimension']),
                           ortho_weight(options['dimension'])], axis=1)
    params['lstm_U'] = lstm_U
    lstm_b = numpy.zeros((4 * options['dimension'],))
    params['lstm_b'] = lstm_b.astype(config.floatX)  # @UndefinedVariable
    w = numpy.random.rand(options['dimension'], ) 
    params['w']=w.astype(config.floatX)  # @UndefinedVariable
    
    return params
    
    
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

main_dir='D:/dataset/test/icde2016_metagraph/'
def proxEmbedTraining(
                     trainingDataFile=main_dir+'facebook.splits/train.10/train_classmate_1', # the full path of training data file
                     wordsEmbeddings=None, # words embeddings
                     wordsEmbeddings_path=main_dir+'facebook/nodesFeatures', # the file path of words embeddings
                     word_dimension=22, # dimension of words embeddings
                     dimension=64, # the dimension of paths embeddings
                     wordsSize=1000000, # the size of words vocabulary
                     subpaths_map=None, # contains sub-paths
                     subpaths_file=main_dir+'facebook/subpathsSaveFile',# the file which contains sub-paths
                     maxlen_subpaths=1000, # the max length for sub-paths
                     h_output_method='mean-pooling', # the output way of lstm. There are three ways, "h" only uses the last output h as the output of lstm for one path; "mean-pooling" uses the mean-pooling of all hi as the output of lstm for one path; "max-pooling" uses the max-pooling of all hi as the output of lstm for one path.
                     maxlen=100,  # Sequence longer then this get ignored 
                     batch_size=1, # use a batch for training. This is the size of this batch.
                     is_shuffle_for_batch=False, # if need shuffle for training
                     discount_alpha=0.1, # the parameter alpha for discount. The longer the subpath, the little will the weight be.
                     subpaths_pooling_method='max-pooling', # the ways to combine several subpaths to one. "mean-pooling" means to combine all subpaths to one by mean-pooling; "max-pooling" means to combine all subpaths to one by max-pooling.
                     objective_function_method='hinge-loss', # loss function, we use sigmoid
                     objective_function_param=0, # the parameter in loss function, beta
                     lrate=0.0001, # learning rate
                     max_epochs=10, # the max epochs for training
                     
                     dispFreq=5, # the frequences for display
                     saveFreq=5, # the frequences for saving the parameters
                     saveto=main_dir+'facebook/proxEmbed-modelParams.npz', # the path for saving parameters. It is generated by main_dir, dataset_name, suffix, class_name and index.
                     
                     # the normalization of this model, l2-norm of all parameters
                     decay_lstm_W=0.01, 
                     decay_lstm_U=0.01, 
                     decay_lstm_b=0.01,
                     decay_w=0.01, 
                     
                     ):
    """
    The training stage of ProxEmbed
    """
    model_options = locals().copy()
    
    if wordsEmbeddings is None: 
        if wordsEmbeddings_path is not None: 
            wordsEmbeddings,dimension,wordsSize=dataProcessTools.getWordsEmbeddings(wordsEmbeddings_path)
        else: 
            print 'There is not path for wordsEmbeddings, exit!!!'
            exit(0) 
    
    if subpaths_map is None:
        if subpaths_file is not None: 
            subpaths_map=dataProcessTools.loadAllSubPaths(subpaths_file, maxlen_subpaths)
        else: 
            print 'There is not path for sub-paths, exit!!!'
            exit(0)
    
    trainingData,trainingPairs=dataProcessTools.getTrainingData(trainingDataFile)
    allBatches=dataProcessTools.get_minibatches_idx(len(trainingData), batch_size, is_shuffle_for_batch)

    params=init_sharedVariables(model_options) 
    tparams=init_tparams(params) 
    print 'Generate models ......'
    
    trainingParis, subPaths_matrix, subPaths_mask, subPaths_lens, wemb, cost=proxEmbedModelMulti.proxEmbedModel(model_options, tparams)
    
    print 'Generate gradients ......'
    grads=tensor.grad(cost,wrt=list(tparams.values()))
    print 'Using Adadelta to generate functions ......'
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update=adadelta(lr, tparams, grads, trainingParis, subPaths_matrix, subPaths_mask, subPaths_lens, wemb, cost)
    
    print 'Start training models ......'
    best_p = None 
    history_cost=[] 
    
    models_count=[0,0,0,0] 
    
    start_time = time.time() 
    print 'start time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    uidx=0 
    for eidx in range(max_epochs):
        for _, batch in allBatches: 
            uidx += 1
            trainingDataForBatch=[trainingData[i] for i in batch]
            trainingPairsForBatch=[trainingPairs[i] for i in batch]
            triples_matrix_data, subPaths_matrix_data, subPaths_mask_data, subPaths_lens_data=dataProcessTools.prepareDataForTraining(trainingDataForBatch, trainingPairsForBatch, subpaths_map)
            cost=0
            cost=f_grad_shared(triples_matrix_data, subPaths_matrix_data, subPaths_mask_data, subPaths_lens_data,wordsEmbeddings)
            f_update(lrate)
            
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                return 
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch =', eidx, ',  Update =', uidx, ',  Cost =', cost
                print 'models_count ==',models_count
            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')
                if best_p is not None: 
                    params = best_p
                else: 
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_cost, **params)
                pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print('Done')
    end_time = time.time() 
    print 'end time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time))
    print 'Training finished! Cost time == ', end_time-start_time,' s'
            
    
if __name__=='__main__':
    print 'Start running proxEmbedTraining......'
    proxEmbedTraining()