from __future__ import absolute_import
from __future__ import division
from locale import currency
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
from code.data.label_gen import Labeller
import logging
import sys
import pickle
logger = logging.getLogger()

class Episode(object):
    #rwd flag turns on the old reward from the original paper with only 1 right answer
    def __init__(self, graph, data, params, rwd):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        # we have correct path passed to the episode, which we will access later
        if rwd:
            start_entities, query_relation,  end_entities, all_answers = data
        else:
            start_entities, query_relation,  all_answers, self.correct_path = data
        self.no_examples = start_entities.shape[0]   #256
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        #turns start entities into a list of the same start entity num_rollouts number of times so we can keep track of all the rollouts later on
        start_entities = np.repeat(start_entities, self.num_rollouts)#KEY LINE RIGHT HERE
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        if rwd:
            end_entities = np.repeat(end_entities, self.num_rollouts)
        all_answers = np.repeat(all_answers, self.num_rollouts)
        self.start_entities = start_entities
        if rwd:
            self.end_entities = end_entities
            self.all_end_entities = all_answers
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        if rwd:
            next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, rwd, self.end_entities)
        else:
            next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, rwd)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        self.rwd = rwd

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self, rwd):
        #instead of rewarding if the exact right correct answer is hit, reward if any correct answer is hit
        if self.rwd:
            reward = (self.current_entities == self.end_entities)
        else:
            reward = []
            for i in range(self.current_entities.shape[0]):
                reward+=[True if self.current_entities[i] in self.all_answers[i] else False]
                #reward+=[True if self.current_entities[i] in self.all_end_entities[i] else False]
            reward = np.array(reward)
        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward

    def backtrack(self, batch):
        # returns all the actions which, taken at the current state, will take the agent to its previous state
        # this allows the agent to learn to backtrack when it makes a mistake
        return np.where(self.state['next_entities'][batch, :] == self.last_entities[batch])[0]

    def num_connections(self, batch):
        pass

    def __call__(self, action):
        #increment path length by 1
        self.current_hop += 1
        self.last_entities=self.current_entities
        #update current entities to be equal to the node the action taken leads to
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]
        #print("Arrived at:")
        #print(self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action])

        #use this new information to generate the next set of possible actions
        if self.rwd:
            next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.rwd, self.end_entities)
        else:
            next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.rwd)

        #and create a new state
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state

import csv

class env(object):
    def __init__(self, params, rwd, mode='train'):
        print("creating environment")
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        self.rwd = rwd
        input_dir = params['data_input_dir']
        #create the whole graph which the agent will traverse along, allowing us to know what actions are possible next
        self.grapher = RelationEntityGrapher(triple_store=params['dataset']['graph'],
                                              max_num_actions=params['max_num_actions'],
                                              entity_vocab=params['entity_vocab'],
                                              relation_vocab=params['relation_vocab'])

        #create the batchers, which in turn create arrays for all triples and query answers. We call these to give us batches
        if mode == 'train':
            self.batcher = RelationEntityBatcher(dataset=params['dataset'],
                                                  batch_size=params['batch_size'],
                                                  entity_vocab=params['entity_vocab'],
                                                  relation_vocab=params['relation_vocab'],
                                                  rwd=rwd)
        else:
            self.batcher = RelationEntityBatcher(dataset=params['dataset'],
                                                  batch_size=params['batch_size'],
                                                  entity_vocab=params['entity_vocab'],
                                                  relation_vocab=params['relation_vocab'],
                                                  rwd=rwd,
                                                  mode=mode)
        if self.rwd:
            self.total_no_examples = self.batcher.store.shape[0]
        #originally max num actions but will be expanded
        self.action_len = self.grapher.array_store.shape[1]
        #creates the filepath of the existing or yet to be generated correct labels csv
        self.correct_filepath = "labels/"+params['dataset_name']+"_labeldict_allact_"+str(params['max_num_actions'])
        #creates the labeller for the environment, which will find the best path by brute force
        self.labeller = Labeller([self.grapher.array_store, params['entity_vocab']['PAD'], params['relation_vocab']['PAD'], params['label_gen'], self.correct_filepath, self.batcher.store_all_correct, rwd, params['random_masking_coef']])

    #returns an episode, a tool which the trainer can use to get current states from, take a step, and then give the actions back to to get another current state until we reach the end
    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train(self.labeller):
                yield Episode(self.grapher, data, params, self.rwd)
        else:
            for data in self.batcher.yield_next_batch_test(self.labeller):
                if data == None:
                    return
                yield Episode(self.grapher, data, params, self.rwd)
