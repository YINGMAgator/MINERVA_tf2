# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jan 20 22:00:11 2022

# @author: yingma, owen burns
# """
# import csv
import numpy as np
import csv
import pickle
import random

class Labeller(object):
    def __init__(self,params):
        self.array_store, self.ePAD, self.rPAD, self.correct_filepath, self.all_correct, self.rwd, self.masking = params
        self.correct={}

    def save_labels(self):
        with open(self.correct_filepath,'wb') as lf:
            pickle.dump(self.correct, lf)

    def get_random_rl_masking(self, line):
        e1=line[0]
        r=line[1]
        return random.sample(self.all_correct[(e1, r)], int(len(self.all_correct[(e1, r)])*self.masking))

    def correct_path(self, line):
        # if this key is already in the dict, dont generate it again. Otherwise, generate the new key
        if self.rwd:
            if self.arr_2_key(line) in list(self.correct.keys()):
                return self.correct[self.arr_2_key(line)]
            else:
                e1 = line[0]
                r = line[1]
                e2 = line[2]
                key=self.arr_2_key(line)
                for path in self.correct_path_generate([e1,r,e2]):
                    if key in self.correct:
                        self.correct[key][0]["N/A"] += [path[0]]
                        if path[0] in self.correct[key][1]:
                            self.correct[key][1][path[0]] += [path[1]]
                        else:
                            self.correct[key][1][path[0]] = [path[1]]
                        if path[1] in self.correct[key][2]:
                            self.correct[key][2][path[1]] += [path[2]]
                        else:
                            self.correct[key][2][path[1]] = [path[2]]
                    else:
                        self.correct[key] = {
                            0: {"N/A" : [path[0]]},
                            1: {path[0] : [path[1]]},
                            2: {path[1] : [path[2]]}
                        }
                return self.correct[key]
        else:
            if self.arr_2_key(line) in list(self.correct.keys()):
                return self.correct[self.arr_2_key(line)]
            else:
                e1=line[0]
                r=line[1]
                key=self.arr_2_key(line)
                for e2 in self.all_correct[(e1, r)]:
                    for path in self.correct_path_generate([e1,r,e2]):
                        if key in self.correct:
                            self.correct[key][0]["N/A"] += [path[0]]
                            if path[0] in self.correct[key][1]:
                                self.correct[key][1][path[0]] += [path[1]]
                            else:
                                self.correct[key][1][path[0]] = [path[1]]
                            if path[1] in self.correct[key][2]:
                                self.correct[key][2][path[1]] += [path[2]]
                            else:
                                self.correct[key][2][path[1]] = [path[2]]
                        else:
                            self.correct[key] = {
                                0: {"N/A" : [path[0]]},
                                1: {path[0] : [path[1]]},
                                2: {path[1] : [path[2]]}
                            }
                return self.correct[key]

    def arr_2_key(self,arr):
        if self.rwd:
            return str(arr[0])+str(arr[1])+str(arr[2])
        else:
            return str(arr[0])+str(arr[1])
        
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
        e1 = line[0]
        r = line[1]
        e2 = line[2]

        # counts how many paths have been returned so far
        paths=0
        # ret1 = all possible first actions
        ret1 = self.array_store[e1, :, :].copy()
        # mask the relation that exactly matches the one in the query and points to the correct answer because it won't teach the agent to look for logical paths
        ret1 = self.mask_out_right_answer(ret1,r,e2)

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

        ################################################
        ####check if the answer is in the second hop####
        ################################################
        # gets all the possible nodes that could be the second hop
        start_entity_2nd_hop = set(self.array_store[e1, :, 0])
        # we don't want the agent to stay on the starting entity
        start_entity_2nd_hop.remove(e1)
        # removes the nodes that were masked last time; if there was a valid one-hop path it would have triggered the if, so this if will only remove previously masked values
        # this handles the case that there is a direct connection btwn node 1 and the answer, which means the answer would show up as a starting node in the second hop
        if e2 in start_entity_2nd_hop:
                start_entity_2nd_hop.remove(e2)
        # temp_paths=paths
        for e21 in start_entity_2nd_hop:
            # ret2 = every possible second action, given the first action
            ret2 = self.array_store[e21, :, :].copy()
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
            ###############################################
            ####check if the answer is in the third hop####
            ###############################################
            # all possible next states given 
            start_entity_3rd_hop= set(self.array_store[e21, :, 0])
            # we don't want the agent to stay on the starting entity
            if e1 in start_entity_3rd_hop:
                start_entity_3rd_hop.remove(e1)
            for e31 in start_entity_3rd_hop:
                #ret3 = every possible third action, given the second action
                ret3 = self.array_store[e31, :, :].copy()
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