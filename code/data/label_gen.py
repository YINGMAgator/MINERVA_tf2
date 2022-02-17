# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jan 20 22:00:11 2022

# @author: yingma, owen burns
# """
# import csv
import numpy as np
import csv

class Labeller(object):
    def __init__(self,params):
        self.array_store, self.ePAD, self.rPAD, self.generate, self.correct_filepath = params
        if not self.generate:
            print("loading correct labels")
            self.correct={}
            with open(self.correct_filepath,'r',newline='') as csvfile:
                reader=csv.reader(csvfile, dialect='excel')
                for x in reader:
                    self.correct[self.arr_2_key(x[:3])]=x[3:]

    def correct_path(self, line):
        if self.generate:
            return self.correct_path_generate(line)
        return [int(x) for x in self.correct[self.arr_2_key(line)]]

    def arr_2_key(self,arr):
        return str(arr[0])+str(arr[1])+str(arr[2])
        
    # generate label for the training data
    def mask_out_right_answer(self, ret,query_relations,answers):
        # ret is all the actions and transitions at the state e1
        relations = ret[:, 1]
        entities = ret[:, 0]
        # true only for a case of an action exactly matching the query relation and answer
        mask = np.logical_and(relations == query_relations, entities == answers)
        # puts masking values on the right answers
        ret[:, 0][mask] = self.ePAD
        ret[:, 1][mask] = self.rPAD
        return ret

    def correct_path_generate(self, line):
        #returns the indexes of the correct actions for each step of the path
        label=[]
        e1 = line[0]
        r = line[1]
        e2 = line[2]
        
        ret1 = self.array_store[e1, :, :].copy()
        # hides the answer with the exactly correct relation and destination entity right out the gate, requiring the algorithm to find a logical path
        ret1 = self.mask_out_right_answer(ret1,r,e2)
        ans_from_1hop = False
        ans_from_2hop = False
        ans_from_3hop = False
        if e2 in ret1[:, 0]:
            # answer exists in 1st hop
            #path is to make the 1 hop then stay in place for the remaining time
            label.append([np.where(ret1[ :, 0]== e2),[np.array([0])],[np.array([0])]])
            # print("Correct Path:")
            # print(ret1[np.where(ret1[ :, 0]== e2)])
            return self.format_return(label)
        else:
            # try find answer in 2nd hop
            start_entity_2nd_hop= set(self.array_store[e1, :, 0])
            # removes the nodes that were masked last time; if there was a valid one-hop path it would have triggered the if, so this if will only remove previously masked values
            if e2 in start_entity_2nd_hop:
                start_entity_2nd_hop.remove(e2)
            # for every possible next entity from the start entity
            for e21 in start_entity_2nd_hop:
                # get every possible action at that entity
                ret2 = self.array_store[e21, :, :].copy()
                # if we are still at the starting entity, again mask out the obvious path
                if e21 == e1:
                    ret2 = self.mask_out_right_answer(ret2,r,e2)
                # if there is an action that takes us to e2, we have found our answer
                if e2 in ret2[:, 0]:
                    # answer exists in 2nd hop
                    # record the steps of the correct path
                    hop1 = np.where(ret1[ :, 0]== e21)   
                    hop2=  np.where(ret2[ :, 0]== e2)
                    #correct path is to make the two hops and then stay in place      
                    label.append([hop1,hop2,[np.array([0])]])
                    # print("Correct Path:")
                    # print(ret1[hop1,0])
                    # print(ret2[hop2,0])
                    return self.format_return(label)
            if not ans_from_2hop:
                #try to find answer in 3rd hop
                #get all possible second nodes, remove e1 (ignore the no_op action)
                start_entity_2nd_hop= set(self.array_store[e1, :, 0])
                start_entity_2nd_hop.remove(e1)
                #remove masked values
                if e2 in start_entity_2nd_hop:
                    start_entity_2nd_hop.remove(e2)
                #for every possible next entity from the start entity
                for e21 in start_entity_2nd_hop:
                    #get the actions at that entity
                    ret2 = self.array_store[e21, :, :].copy()
                    #and all the possible next states
                    start_entity_3nd_hop= set(self.array_store[e21, :, 0])
                    #for every possible third state
                    for e31 in start_entity_3nd_hop:
                        #get all possible actions
                        ret3 = self.array_store[e31, :, :].copy()
                        #if we have moved back to the start state, mask out the obvious path
                        if e31 == e1:
                            ret3 = self.mask_out_right_answer(ret3,r,e2)   
                        if e2 in ret3[:, 0]:
                            # answer exists in 3rd hop
                            hop1 = np.where(ret1[ :, 0]== e21) 
                            hop2=  np.where(ret2[ :, 0]== e31) 
                            hop3=  np.where(ret3[ :, 0]== e2)        
                            label.append([hop1, hop2,hop3])
                            # print("Correct Path:")
                            # print(ret1[hop1,0])
                            # print(ret2[hop2,0])
                            # print(ret3[hop3,0])
                            return self.format_return(label)
        #the 0.08% chance that there is no path found within 3 hops we return []
        return self.format_return(label)

    def format_return(self, label):
        if label == []:
            return np.array([-1,-1,-1])
        else:
            return np.array([label[0][i][0][0] for i in range(len(label[0]))])