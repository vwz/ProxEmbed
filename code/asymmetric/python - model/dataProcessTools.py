#encoding=utf-8
'''
methods for processing data
'''

import numpy
import theano

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def getTrainingData(trainingDataFile):
    '''
        read training data from file
    :type string
    :param trainingDataFile
    '''
    data=[] 
    pairs=[] 
    with open(trainingDataFile) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:
                continue
            arr=[]
            arr.append(tmp[0]+'-'+tmp[1])
            arr.append(tmp[1]+'-'+tmp[0])
            arr.append(tmp[0]+'-'+tmp[2])
            arr.append(tmp[2]+'-'+tmp[0])
            pairs.append(arr) 
            tmp=[int(x) for x in tmp]
            data.append(tmp)
            
    return data,pairs

def getWordsEmbeddings(wordsEmbeddings_path):
    """
        read words embeddings from file
            a b
            c d e f ....
            g h j k ....
            a means the num(line) of the data，b means the dimension of the data
            c and g are the index of the corresponding words
            d，e，f，h，j，k，... are the content of embeddings
    :type String
    :param wordsEmbeddings_path
    """
    size=0
    dimension=0
    wemb=[]
    with open(wordsEmbeddings_path) as f:
        for l in f:
            arr=l.strip().split()
            if len(arr)==2:
                size=int(arr[0])
                dimension=int(arr[1])
                wemb=numpy.zeros((size,dimension)) 
                continue
            id=int(arr[0]) 
            for i in range(0,dimension):
                wemb[id][i]=float(arr[i+1])
    return wemb,dimension,size

def loadAllSubPaths(subpaths_file,maxlen=1000):
    """
        read all subpaths from file
    :type subpaths_file: String
    :param subpaths_file：file path 
       
    :type maxlen:int
    :param maxlen:
    
    the return value is a map, and the key of this map is made of startNodeId-endNodeId.
    the value of this map is a list made of startNodeId aId bId cId dId... endNodeId
    """
    map={}
    with open(subpaths_file) as f:
        for l in f: 
            splitByTab=l.strip().split('\t')
            key=splitByTab[0]+'-'+splitByTab[1] 
            sentence=[int(y) for y in splitByTab[2].split()[:]] 
            if len(sentence)>maxlen: 
                continue
            if key in map: 
                map[key].append(sentence)
            else: 
                tmp=[]
                tmp.append(sentence)
                map[key]=tmp
    return map

def prepareDataForTraining(trainingDataTriples,trainingDataPairs,subpaths_map):
    """
        prepare data for training
    """
    n_triples=len(trainingDataTriples)
    
    triples_matrix=numpy.zeros([n_triples,4,2]).astype('int64')
    
    maxlen=0 
    n_subpaths=0 
    allPairs=[] 
    for list in trainingDataPairs:
        for l in list:
            allPairs.append(l)
    for key in allPairs: 
        if key not in subpaths_map: 
            continue;
        list=subpaths_map[key]
        n_subpaths+=len(list) 
        for l in list: 
            if len(l)>maxlen:
                maxlen=len(l)
                
    subPaths_matrix=numpy.zeros([maxlen,n_subpaths]).astype('int64') 
    
    subPaths_mask=numpy.zeros([maxlen,n_subpaths]).astype(theano.config.floatX)  # @UndefinedVariable
    
    subPaths_lens=numpy.zeros([n_subpaths,]).astype('int64')
    
    current_index=0 
    path_index=0 
    valid_triples_count=0 
    for i in range(len(trainingDataPairs)): 
        pairs=trainingDataPairs[i] 
        
        valid_triples_count+=1 
        for j in range(len(pairs)): 
            pair=pairs[j]
            list=None
            if pair in subpaths_map: 
                list=subpaths_map[pair] 
            if list is not None:
                triples_matrix[i][j][0]=current_index 
                current_index+=len(list)
                triples_matrix[i][j][1]=current_index
                for x in range(len(list)):
                    index=path_index+x 
                    path=list[x] 
                    subPaths_lens[index]=len(path) 
                    for y in range(len(path)): 
                        subPaths_matrix[y][index]=path[y]
                        subPaths_mask[y][index]=1. 
                path_index+=len(list) 
            else : 
                triples_matrix[i][j][0]=current_index 
                current_index+=0
                triples_matrix[i][j][1]=current_index 
    
    count=0
    for i in range(len(triples_matrix)):
        if triples_matrix[i][0][0]!=triples_matrix[i][1][1] and triples_matrix[i][2][0]!=triples_matrix[i][3][1]:
            count+=1
    triples_matrix_new=numpy.zeros([count,4,2]).astype('int64')
    index=0
    for i in range(len(triples_matrix)):
        if triples_matrix[i][0][0]!=triples_matrix[i][1][1] and triples_matrix[i][2][0]!=triples_matrix[i][3][1]:
            triples_matrix_new[index]=triples_matrix[i]
            index+=1
    triples_matrix=triples_matrix_new
    
    return triples_matrix, subPaths_matrix, subPaths_mask, subPaths_lens
    
    
def prepareDataForTest(query,candidate,subpaths_map):
    """
   prepare data for test
    """
    key1=bytes(query)+'-'+bytes(candidate)
    key2=bytes(candidate)+'-'+bytes(query)
    if key1 not in subpaths_map and key2 not in subpaths_map:
        return None,None,None
    subpaths=[]
    if key1 in subpaths_map:
        subpaths.extend(subpaths_map[key1]) 
    if key2 in subpaths_map:
        subpaths.extend(subpaths_map[key2]) 
    maxlen=0
    for subpath in subpaths:
        if len(subpath)>maxlen:
            maxlen=len(subpath)
    subPaths_matrix=numpy.zeros([maxlen,len(subpaths)]).astype('int64')
    subPaths_mask=numpy.zeros([maxlen,len(subpaths)]).astype(theano.config.floatX)  # @UndefinedVariable
    subPaths_lens=numpy.zeros([len(subpaths),]).astype('int64')
    for i in range(len(subpaths)): 
        subpath=subpaths[i] 
        subPaths_lens[i]=len(subpath) 
        for j in range(len(subpath)):
            subPaths_matrix[j][i]=subpath[j] 
            subPaths_mask[j][i]=1. 
    
    return subPaths_matrix,subPaths_mask,subPaths_lens
            
            
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


