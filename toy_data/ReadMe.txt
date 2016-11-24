This directory contains a toy data set for ProxEmbed model.
The toy data set is constructed according to the Figure 1 in the following paper:

@inproceedings{LiuZZZCWY17,
 author = {Liu, Zemin and Zheng, Vincent W. and Zhao, Zhou and Zhu, Fanwei and Chang, Kevin Chen-Chuan and Wu, Minghui and Ying, Jing},
 title = {Semantic Proximity Search on Heterogeneous Graph by Proximity Embedding},
 booktitle = {Proc. of the 31st AAAI Conference on Artificial Intelligence},
 series = {AAAI '17},
 year = {2017}
} 

Please cite the above reference for using our code and data.
========================================================================================================

1. In each folder 'dataset' (where 'dataset' = linkedin or dblp)

File "graph.node" : The nodes in each graph. Each row has 3 columns: column 1 is node ID, column 2 is node type, column 3 is node value.(we only need to use column 1 and 2)

File "graph.edge" : The edges in each graph. Each row indicates a directed edge.

******************************************

2. In each folder 'dataset.splits' (where 'dataset' = linkedin or dblp)

2.1. In sub-folder 'train.labelSize' (where 'labelSize' = 4, 100 or 1000) (while in our real dataset 'labelSize' = 10, 100 or 1000)

File "train_relation_splitId" : One split of training data for a relation. 'splitId' is from 1 to 3, thus we will train our model on 3 different training data sets. (while in our real dataset 'splitId' is from 1 to 10)

Example : for "./linkedin.splits/train.4/train_school_1" 
a) we have 4 labels, where each label is a tuple of <queryNodeId, targetNodeId_1, targetNodeId_2>, meaning: given queryNodeId, targetNodeId_1 is closer to queryNode than targetNodeId_2.
b) it is the 1st split of training data for relation 'school'.

We do not provide the 'labelSize' = 100 and 1000, becasue the toydata is too small.

2.2. In sub-folder 'test'

File "test_relation_splitId" : One split of test data for a relation. 

Example : for "./linkedin.splits/test/test_school_1"
a) it is the 1st split of test data for relation 'school', thus used by all the "./linkedin.splits/train.labelSize/train_school_1" files (where 'labelSize' = 4, 100 or 1000).(while in our real dataset 'labelSize' = 10, 100 or 1000)
b) each line of the file is <queryNodeId, targetNodeId_1, targetNodeId_2, ..., targetNodeId_m>, meaning: given queryNodeId, we want to apply our model and generate a ranking list over the targetNodeId_1, targetNodeId_2 until targetNodeId_m. For different lines, m can be different. 


2.3. In sub-folder 'ideal'

File "ideal_relation_splitId" : One split of ideal ranking for a relation. 

Example : for "./linkedin.splits/ideal/ideal_school_1" 
a) it is the 1st split of ideal ranking for relation 'school', which is used to evaluate the ranking prediction for "./linkedin.splits/test/test_school_1";
b) each line of the file is <queryNodeId, targetNodeId_g1, targetNodeId_g2, ..., targetNodeId_gn,>, meaning: given queryNodeId, the ground truth relevant target nodes are targetNodeId_g1, targetNodeId_g2, ..., targetNodeId_gn. Note that gn can be different from m; i.e., for the queryNodeId, there are m test targetNodeId's for us to rank, but in the end only a subset of them are relevant by ground truth. 



