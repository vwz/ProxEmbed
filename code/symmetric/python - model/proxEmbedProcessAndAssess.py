#encoding=utf-8
'''
process dataset by proxEmbed model and then assess
'''

import numpy
import theano
from theano import tensor
from collections import OrderedDict
import proxEmbedProcessModel
import dataProcessTools
import toolsFunction
import evaluateTools


def load_params(path, params):
    pp = numpy.load(path) 
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def get_path2vecModel(
                      
                   model_params_path='', # the path of model parameters
                     word_dimension=0, # the dimension of words embedding 
                     dimension=0, # the dimension of path embedding
                     h_output_method='h', # the output way of lstm
                     discount_alpha=0.1, # discount alpha
                     subpaths_pooling_method='max-pooling', # the combine way of sub-paths
                      ):
    """
    get model from file
    """
    model_options = locals().copy()
    
    tparams = OrderedDict()
    tparams['lstm_W']=None
    tparams['lstm_U']=None
    tparams['lstm_b']=None
    tparams['w']=None
    tparams=load_params(model_params_path, tparams)
    
    subPaths_matrix,subPaths_mask,subPaths_lens,wemb,score=proxEmbedProcessModel.proxEmbedModel(model_options, tparams)
    func=theano.function([subPaths_matrix,subPaths_mask,subPaths_lens,wemb], score) 
    
    return func 


def compute_path2vec(
                     wordsEmbeddings=None, # words embeddings
                     wordsEmbeddings_path=None, # the file path of words embeddings
                     word_dimension=0, #  dimension of words embeddings
                     dimension=0, # the dimension of paths embeddings
                     wordsSize=0, # the size of words vocabulary
                     subpaths_map=None, # contains sub-paths
                     subpaths_file=None,# the file which contains sub-paths
                     maxlen_subpaths=1000, # the max length for sub-paths
                     maxlen=100,  # Sequence longer then this get ignored 
                     
                     test_data_file='', # the file path of test data
                     top_num=10, # the top num to predict
                     ideal_data_file='', # ground truth
                     func=None, # function
                   ):
    """
    compute the result of the model
    """
    
    model_options = locals().copy()
    
    if wordsEmbeddings is None: 
        if wordsEmbeddings_path is not None: 
            wordsEmbeddings,dimension,wordsSize=dataProcessTools.getWordsEmbeddings(wordsEmbeddings_path)
        else: 
            print 'There is not path for wordsEmbeddings, exit！！！'
            exit(0) 

    if subpaths_map is None: 
        if subpaths_file is not None: 
            subpaths_map=dataProcessTools.loadAllSubPaths(subpaths_file, maxlen_subpaths)
        else: 
            print 'There is not path for sub-paths, exit！！！'
            exit(0)

    line_count=0 
    test_map={} 
    print 'Compute MAP and nDCG for file ',test_data_file
    with open(test_data_file) as f: 
        for l in f: 
            arr=l.strip().split()
            query=int(arr[0]) 
            map={} 
            for i in range(1,len(arr)):
                candidate=int(arr[i]) 
                subPaths_matrix_data,subPaths_mask_data,subPaths_lens_data=dataProcessTools.prepareDataForTest(query, candidate, subpaths_map)
                if subPaths_matrix_data is None and subPaths_mask_data is None and subPaths_lens_data is None: 
                    map[candidate]=-1000. 
                else: 
                    value=func(subPaths_matrix_data,subPaths_mask_data,subPaths_lens_data,wordsEmbeddings)
                    map[candidate]=value
            
            tops_in_line=toolsFunction.mapSortByValueDESC(map, top_num)
            test_map[line_count]=tops_in_line 
            line_count+=1 
                
    line_count=0 
    ideal_map={}
    with open(ideal_data_file) as f: 
        for l in f: 
            arr=l.strip().split()
            arr=[int(x) for x in arr] 
            ideal_map[line_count]=arr[1:] 
            line_count+=1 
    
    MAP=evaluateTools.get_MAP(top_num, ideal_map, test_map)
    MnDCG=evaluateTools.get_MnDCG(top_num, ideal_map, test_map)
    
    return MAP,MnDCG
    
    
    
    
    