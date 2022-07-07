#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:03:03 2022

@author: yingma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 22:00:11 2022

@author: yingma
"""
import csv
import numpy as np
from code.data.vocab_gen import Vocab_Gen
from collections import defaultdict
from collections import Counter
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

def check_correctness(corr,PATH):
    cnt = 0
    cnt1=0     
    for i in corr[0]['N/A']:
        #print(i)
        e_1 = array_store[e1, i, 0]
        for j in corr[1][i]:
            e_2 =array_store[e_1, j, 0]
            
            for k in corr[2][j]:
                cnt+=1
                e_3 = array_store[e_2, k, 0]
                if e_3==e2:
                    b=0
                    cnt1+=1
                else:
                    b=1
                if [i,j,k] in PATH:
                    a=0
                else:
                    a=1
    print('there are', cnt1/cnt, 'correct paths',cnt1,cnt)
                
    return a,b,cnt1/cnt

def check_correctness_updated(corr,PATH):
    cnt = 0
    cnt1=0    
    arr_1=[]
    arr_2=[]
    arr_3=[]
    for i in corr[0]['N/A']:
        #print(i)
        if not i in arr_1:
            
            e_1 = array_store[e1, i, 0]
            
        for j in corr[1][i]:
            e_2 =array_store[e_1, j, 0]

            for k in corr[2][tuple([i,j])]:
                cnt+=1
                e_3 = array_store[e_2, k, 0]
                if e_3==e2:
                    b=0
                    cnt1+=1
                else:
                    b=1
                if [i,j,k] in PATH:
                    a=0
                else:
                    a=1
                    #print('ijk',i,j,k)
    print('there are', cnt1/cnt, 'correct paths',cnt1,cnt)
                
    return a,cnt1,cnt1/cnt


def correct_path_updated(line):
     # if this key is already in the dict, dont generate it again. Otherwise, generate the new key
     

    e1 = line[0]
    r = line[1]
    e2 = line[2]
    correct={}
    first = False
    PATH=[]
 
    for path in correct_path_generate(line):
        PATH.append(list(path))
#                print(self.neg_cnt/self.cnt)
        e01=tuple([path[0],path[1]]) 
        if first:
            #if not path[0] in correct[0]["N/A"]:
            correct[0]["N/A"] += [path[0]]
            if path[0] in correct[1]:
             #   if not path[1] in correct[1][path[0]]:
                 correct[1][path[0]] += [path[1]]
            else:
                correct[1][path[0]] = [path[1]]
            
            if e01 in correct[2]:
              #  if not path[2] in correct[2][e01]:
                correct[2][e01] += [path[2]]
            else:
                correct[2][e01] = [path[2]]
        else:
            correct= {
                0: {"N/A" : [path[0]]},
                1: {path[0] : [path[1]]},
                2: {e01 : [path[2]]}
            }
            first=True
        
        a,b,c=check_correctness_updated(correct,PATH)
        if b!=len(PATH):
            print(len(PATH),PATH)
            print(b,correct )
            break
        # if c!=1:
        #     break
    return correct,PATH

         
def correct_path(line):
     # if this key is already in the dict, dont generate it again. Otherwise, generate the new key
     

    e1 = line[0]
    r = line[1]
    e2 = line[2]
    correct={}
    first = False
    PATH=[]
 
    for path in correct_path_generate(line):
        PATH.append(list(path))
#                print(self.neg_cnt/self.cnt)

        if first:
            correct[0]["N/A"] += [path[0]]
            if path[0] in correct[1]:
                correct[1][path[0]] += [path[1]]
            else:
                correct[1][path[0]] = [path[1]]
            if path[1] in correct[2]:
                correct[2][path[1]] += [path[2]]
            else:
                correct[2][path[1]] = [path[2]]
        else:
            correct= {
                0: {"N/A" : [path[0]]},
                1: {path[0] : [path[1]]},
                2: {path[1] : [path[2]]}
            }
            first=True
        a,b,c=check_correctness(correct,PATH)
        
        print(PATH)
        print(correct )
        if c!=1:
            break
    return correct,PATH

     
 # generate label for the training data
def mask_out_right_answer(ret,query_relations,answers):
    # ret is all the actions and transitions at the state e1
    relations = ret[:, 1]
    entities = ret[:, 0]
    # true only for a case of an action exactly matching the query relation and answer
    mask = np.logical_and(relations == query_relations, entities == answers)
    # puts masking values on the right answers
    ret[:, 0][mask] = ePAD
    ret[:, 1][mask] = rPAD
    return ret

def correct_path_generate( line):

    #returns the indexes of the correct actions for each step of the path
    e1 = line[0]
    r = line[1]
    e2 = line[2]

    # counts how many paths have been returned so far
    paths=0
    # ret1 = all possible first actions
    ret1 = array_store[e1, :, :].copy()
    # mask the relation that exactly matches the one in the query and points to the correct answer because it won't teach the agent to look for logical paths
    ret1 = mask_out_right_answer(ret1,r,e2)

    ###############################################
    ####check if the answer is in the first hop####
    ###############################################
    if e2 in ret1[:, 0]:
        # get all indexes (used to identify actions) where the destination node is the correct node
        valid_actions = np.where(ret1[ :, 0]== e2)[0]
        # loop through and yield each relevant path
        for x in valid_actions:
            paths+=1
            yield np.array([x,0,0], int)
      #  print(e2,0,0)
    ################################################
    ####check if the answer is in the second hop####
    ################################################
    # gets all the possible nodes that could be the second hop
    start_entity_2nd_hop = set(array_store[e1, :, 0])
    # we don't want the agent to stay on the starting entity
    start_entity_2nd_hop.remove(e1)
    # removes the nodes that were masked last time; if there was a valid one-hop path it would have triggered the if, so this if will only remove previously masked values
    # this handles the case that there is a direct connection btwn node 1 and the answer, which means the answer would show up as a starting node in the second hop
    if e2 in start_entity_2nd_hop:
            start_entity_2nd_hop.remove(e2)
    # temp_paths=paths
    for e21 in start_entity_2nd_hop:
        # ret2 = every possible second action, given the first action
        ret2 = array_store[e21, :, :].copy()
        # if there is an action that takes us to e2, we have found our answer
        if e2 in ret2[:, 0]:
            # every action that could lead from the first node to the current node
            hop1 = np.where(ret1[ :, 0] == e21)[0]
            # every action that could lead from the current node to the answer
            hop2=  np.where(ret2[ :, 0] == e2)[0]
            for h1 in hop1:
                for h2 in hop2:
                    paths+=1
                    yield np.array([h1, h2, 0], int)
          #  print(e21,e2,0)
        ###############################################
        ####check if the answer is in the third hop####
        ###############################################
        # all possible next states given 
        start_entity_3rd_hop= set(array_store[e21, :, 0])
        # we don't want the agent to stay on the starting entity
        if e1 in start_entity_3rd_hop:
            start_entity_3rd_hop.remove(e1)
        if e2 in start_entity_3rd_hop:
            start_entity_3rd_hop.remove(e2)
        for e31 in start_entity_3rd_hop:
            #ret3 = every possible third action, given the second action
            ret3 = array_store[e31, :, :].copy()
            if e2 in ret3[:, 0]:
                # all actions that take you from the start node to the second node
                hop1 = np.where(ret1[ :, 0]== e21)[0]
                # all actions that take you from the second node to the current node
                hop2=  np.where(ret2[ :, 0]== e31)[0]
                # all actions that takes you from the current node to the answer
                hop3=  np.where(ret3[ :, 0]== e2)[0]
                for h1 in hop1:
                    for h2 in hop2:
                        for h3 in hop3:
                            paths+=1
                            yield np.array([h1, h2, h3], int)
              #  print(e21,e31,e2)
            # else:
            #     #if that third node does not lead to the answer, go back
            #     # all actions that take you from the start node to the second node
            #     hop1 = np.where(ret1[ :, 0]== e21)[0]
            #     # all actions that take you from the second node to the current node
            #     hop2=  np.where(ret2[ :, 0]== e31)[0]
            #     # all actions that takes you from the current node back to the second node
            #     hop3=  np.where(ret3[ :, 0]== e21)[0]
            #     for h1 in hop1:
            #         for h2 in hop2:
            #             for h3 in hop3:
            #                 print(h3)
            #                 paths+=1
            #                 yield np.array([h1, h2, h3], int)
        # # we get here without yielding anything if the second action we took can't lead to a correct answer
        # if paths-temp_paths == 0:
        #     # all actions that take you from the start node to the second node
        #     hop1 = np.where(ret1[ :, 0] == e21)[0]
        #     # all actions that take you from the second node back to the start node
        #     hop2=  np.where(ret2[ :, 0] == e1)[0]
        #     for h1 in hop1:
        #         for h2 in hop2:
        #             paths+=1
        #             print(h2)
        #             yield np.array([h1, h2, 0], int)

    if paths == 0:
        yield np.array([-1, -1, -1], int)
   
label=[]
triple_store = options['dataset']['train']
ans_hop_cnt =[0,0,0]
find_index = 0

for line in triple_store:
    
    
    e1 = entity_vocab[line[0]]
    r = relation_vocab[line[1]]
    e2 = entity_vocab[line[2]]
    corr,PATH=correct_path_updated([e1,r,e2] )
   # print(corr)
 
# unique path generated
 #    print(len(PATH))
 # #   print(PATH)
 #    PATH_uni=[]
    
 #    for p in PATH:
 #        #print(p)
 #        if not p in PATH_uni:
 #            PATH_uni.append(p)
 #        else:
 #            print(p)
 #    print(len(PATH_uni))
    
    a,b,rate=check_correctness_updated(corr,PATH)

    
    # for i in corr[0]['N/A']:
    #     #print(i)
    #     e_1 = array_store[e1, i, 0]
    #     for j in corr[1][i]:
    #         e_2 =array_store[e_1, j, 0]
            
    #         for k in corr[2][j]:
    #             e_3 = array_store[e_2, k, 0]
    #             if e_3==e2:
    #                 cnt+=1 
    #             if [i,j,k] in PATH:
    #                 print('y')
    #             else:
    #                 print('N')
    #             cnt1+=1








