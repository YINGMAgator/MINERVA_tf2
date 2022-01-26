#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 22:00:11 2022

@author: yingma
"""
import csv
import numpy as np
from vocab_gen import Vocab_Gen
from collections import defaultdict
#parameters
max_num_actions = 200


## read in data
options={}
options['dataset']={}
options['data_input_dir']='datasets/data_preprocessed/FB15K-237/'
Dataset_list=['train','test','dev','graph']
for dataset in Dataset_list:
    input_file = options['data_input_dir']+dataset+'.txt'
    ds = []
    with open(input_file) as raw_input_file:
        csv_file = csv.reader(raw_input_file, delimiter = '\t' )
        for line in csv_file:
            ds.append(line)   
    options['dataset'][dataset]=ds
    


##  reading vocab files...
vocab=Vocab_Gen(Datasets=[options['dataset']['train'],options['dataset']['test'],options['dataset']['graph']])
options['relation_vocab'] = vocab.relation_vocab
options['entity_vocab'] = vocab.entity_vocab
    

# save the graph into an array_store
triple_store = options['dataset']['graph']
relation_vocab = options['relation_vocab'] 
entity_vocab = options['entity_vocab']
ePAD = entity_vocab['PAD']
rPAD = relation_vocab['PAD']
store = defaultdict(list)
test_store = defaultdict(list)
array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32'))
array_store[:, :, 0] *= ePAD
array_store[:, :, 1] *= rPAD


for line in triple_store:
    e1 = entity_vocab[line[0]]
    r = relation_vocab[line[1]]
    e2 = entity_vocab[line[2]]
    store[e1].append((r, e2))

for e1 in store:
    num_actions = 1
    array_store[e1, 0, 1] = relation_vocab['NO_OP']
    array_store[e1, 0, 0] = e1
    for r, e2 in store[e1]:
        if num_actions == array_store.shape[1]:
            break
        array_store[e1,num_actions,0] = e2
        array_store[e1,num_actions,1] = r
        num_actions += 1
        
# check whether the answer exist in the graph        
# load the test dataset
# triple_store = options['dataset']['test']
# ans_hop=np.zeros(len(triple_store))
# i=0
# for line in triple_store:
#     e1 = entity_vocab[line[0]]
#     e2 = entity_vocab[line[2]]
#     answer_lib = set(array_store[e1, :, 0])
#     if e2 in answer_lib:
#         ans_hop[i] = 1
#     else:
#         new_answer_lib=set()
#         for e_index in answer_lib:
#             set_sub = array_store[e_index, :, 0]
#             new_answer_lib = new_answer_lib.union(set_sub)
#         answer_lib = new_answer_lib   
#         if e2 in answer_lib:
#             ans_hop[i] = 2
#         else:   
#             new_answer_lib=set()
#             for e_index in answer_lib:
#                 set_sub = array_store[e_index, :, 0]
#                 new_answer_lib = new_answer_lib.union(set_sub)
#             answer_lib = new_answer_lib   
#             if e2 in answer_lib:
#                 ans_hop[i] = 3
#     print(i,ans_hop[i])
#     i = i+1      

# generate label for the training data
def mask_out_right_answer(ret,query_relations,answers):
    relations = ret[:, 1]
    entities = ret[:, 0]
    mask = np.logical_and(relations == query_relations, entities == answers)
    ret[:, 0][mask] = ePAD
    ret[:, 1][mask] = rPAD
    return ret
    
    
label=[]
triple_store = options['dataset']['train']
ans_hop_cnt =[0,0,0]
find_index = 0
cnt = 0
for line in triple_store:
    cnt+=1
    e1 = entity_vocab[line[0]]
    r = relation_vocab[line[1]]
    e2 = entity_vocab[line[2]]
    
    ret1 = array_store[e1, :, :].copy()
    ret1 = mask_out_right_answer(ret1,r,e2)
    ans_from_1hop = False
    ans_from_2hop = False
    ans_from_3hop = False
    if e2 in ret1[:, 0]:
        # answer exists in 1st hop
        label.append(  np.where(ret1[ :, 0]== e2))
        ans_from_1hop = True
        ans_hop_cnt[0]+=1
    else:
        # try find answer in 2nd hop
         
        start_entity_2nd_hop= set(array_store[e1, :, 0])
        if e2 in start_entity_2nd_hop:
            start_entity_2nd_hop.remove(e2)
        for e21 in start_entity_2nd_hop:
            ret2 = array_store[e21, :, :].copy()
            if e21 == e1:
                ret2 = mask_out_right_answer(ret2,r,e2)   
            if e2 in ret2[:, 0]:
                # answer exists in 2nd hop
                hop1 = np.where(ret1[ :, 0]== e21)   
                hop2=  np.where(ret2[ :, 0]== e2)        
                label.append([hop1,hop2])
                ans_from_2hop = True
                ans_hop_cnt[1]+=1
                break
        if not ans_from_2hop:
            #try to find answer in 3rd hop
            start_entity_2nd_hop= set(array_store[e1, :, 0])
            start_entity_2nd_hop.remove(e1)
            if e2 in start_entity_2nd_hop:
                start_entity_2nd_hop.remove(e2)
            for e21 in start_entity_2nd_hop:
                ret2 = array_store[e21, :, :].copy()
                start_entity_3nd_hop= set(array_store[e21, :, 0])
                for e31 in start_entity_3nd_hop:
                    ret3 = array_store[e31, :, :].copy()
                    if e31 == e1:
                        ret3 = mask_out_right_answer(ret3,r,e2)   
                    if e2 in ret3[:, 0]:
                        # answer exists in 2nd hop
                        hop1 = np.where(ret1[ :, 0]== e21) 
                        hop2=  np.where(ret2[ :, 0]== e31) 
                        hop3=  np.where(ret3[ :, 0]== e2)        
                        label.append ([hop1, hop2,hop3])
                        ans_from_3hop = True
                        ans_hop_cnt[2]+=1
                        break
                if ans_from_3hop:
                    break
    if ans_from_1hop or  ans_from_2hop or ans_from_3hop:
        find_index+=1
    else:
        if cnt%100 ==1 :
            print(cnt, 'find answer',round(find_index/cnt,4),
                  '%, find answer in 1st hop', round(ans_hop_cnt[0]/cnt,3),
                  '; 2nd hop', round(ans_hop_cnt[1]/cnt,3),
                  '; 3rd hop', round(ans_hop_cnt[2]/cnt,3),)
                












