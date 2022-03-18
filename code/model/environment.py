from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
from code.data.label_gen import Labeller
import logging
import sys

logger = logging.getLogger()


class Episode(object):

    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        # we have correct path passed to the episode, which we will access later
        start_entities, query_relation,  end_entities, all_answers, self.correct_path = data
        self.no_examples = start_entities.shape[0]   #256
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        #turns start entities into a list of the same start entity num_rollouts number of times so we can keep track of all the rollouts later on
        start_entities = np.repeat(start_entities, self.num_rollouts) 
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward

    def __call__(self, action):
        #increment path length by 1
        self.current_hop += 1
        #update current entities to be equal to the node the action taken leads to
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]
        #print("Arrived at:")
        #print(self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action])

        #use this new information to generate the next set of possible actions
        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts )

        #and create a new state
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state

import csv

class env(object):
    def __init__(self, params, mode='train'):

        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        input_dir = params['data_input_dir']
        #create the batchers, which in turn create arrays for all triples and query answers. We call these to give us batches
        if mode == 'train':
            self.batcher = RelationEntityBatcher(dataset=params['dataset'],
                                                  batch_size=params['batch_size'],
                                                  entity_vocab=params['entity_vocab'],
                                                  relation_vocab=params['relation_vocab']
                                                  )
        else:
            self.batcher = RelationEntityBatcher(dataset=params['dataset'],
                                                  batch_size=params['batch_size'],
                                                  entity_vocab=params['entity_vocab'],
                                                  relation_vocab=params['relation_vocab'],
                                                  mode=mode)

            self.total_no_examples = self.batcher.store.shape[0]
        #create the whole graph which the agent will traverse along, allowing us to know what actions are possible next
        self.grapher = RelationEntityGrapher(triple_store=params['dataset']['graph'],
                                              max_num_actions=params['max_num_actions'],
                                              entity_vocab=params['entity_vocab'],
                                              relation_vocab=params['relation_vocab'])
        #creates the filepath of the existing or yet to be generated correct labels csv
        correct_filepath="C:\\Users\\owenb\\OneDrive\\Documents\\GitHub\\MINERVA_tf2\\labels\\"+params['dataset_name']+"_labels.csv"
        #creates the labeller for the environment, which will find the best path by brute force
        self.labeller = Labeller([self.grapher.array_store, params['entity_vocab']['PAD'], params['relation_vocab']['PAD'], params['label_gen'], correct_filepath])
        #Code to generate labels for all of the potential queries and save them to a CSV file
        if(params['label_gen']):
            print("generating labels file")
            with open(correct_filepath,'w',newline='') as csvfile:
                writer=csv.writer(csvfile, dialect='excel')
                for x in range(len(self.batcher.store)):
                    #writes a row with the query followed by the correct steps
                    for path in self.labeller.correct_path(self.batcher.store[x,:]):
                        writer.writerow(np.concatenate((self.batcher.store[x,:],path)))
            sys.exit("Correct labels written to "+params['dataset_name']+"_labels.csv")


    #returns an episode, a tool which the trainer can use to get current states from, take a step, and then give the actions back to to get another current state until we reach the end
    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train(self.labeller):
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test(self.labeller):
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
