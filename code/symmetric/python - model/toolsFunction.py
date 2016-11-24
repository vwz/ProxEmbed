#encoding=utf-8
'''
some tools methods
'''

import numpy
import theano
from theano import tensor
from theano.ifelse import ifelse

def mapSortByValueDESC(map,top):
    """
    sort by value desc
    """
    if top>len(map): 
        top=len(map)
    items=map.items() 
    backitems=[[v[1],v[0]] for v in items]  
    backitems.sort(reverse=True) 
    e=[ backitems[i][1] for i in range(top)]  
    return e


def mapSortByValueASC(map,top):
    """
    sort by value asc
    """
    if top>len(map): 
        top=len(map)
    items=map.items() 
    backitems=[[v[1],v[0]] for v in items]  
    backitems.sort() 
    e=[ backitems[i][1] for i in range(top)]  
    return e


def max_poolingForMatrix(x):
    """
       abs max pooling method
    """
    def _funcForRow(row,max_array):
        def _funcForElement(element,max_value):
            return ifelse(tensor.gt(tensor.abs_(element), tensor.abs_(max_value)),  element,  max_value)
    
        r,u=theano.scan(
                    fn=_funcForElement,
                    sequences=[row,max_array],
                    )
        return r

    rval,update=theano.scan(
                        fn=_funcForRow,
                        sequences=x,
                        outputs_info=tensor.alloc(numpy.asarray(0., dtype=theano.config.floatX), # @UndefinedVariable
                                                           x.shape[1],
                                                           ),
                        )
    return rval[-1]