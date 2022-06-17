from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os
from code.data.label_gen import Labeller


class RelationEntityBatcher():

    def __init__(self,dataset,batch_size, entity_vocab, relation_vocab, rwd, mode = "train", num_rollouts=20):
        # self.input_dir = input_dir
        # self.input_file = input_dir+'/{0}.txt'.format(mode)
        self.train_data = dataset['train']
        self.test_data = dataset['test']
        self.dev_data = dataset['dev']
        self.graph_data = dataset['graph'] 
        self.batch_size = batch_size
        self.full_graph = dataset['full_graph']
        self.num_rollouts = num_rollouts
        print('Reading vocab...')
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.mode = mode
        self.rwd = rwd
        self.create_triple_store()
        print("batcher loaded")

    #UNDERSTOOD
    def get_next_batch(self):
        if self.mode == 'train':
            yield self.yield_next_batch_train()
        else:
            yield self.yield_next_batch_test()

    #UNDERSTOOD
    def create_triple_store(self):
        self.store_all_correct = defaultdict(set)
        if self.rwd:
            self.store = []

        if self.mode == 'train':
            # with open(input_file) as raw_input_file:
                # csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            #goes through the training data and adds the encoding generated in vocab_gen.py for every fact to a list. Also creates a dictionary that for a query returns a list of correct answers
            #e.g. e1= turkey r= neighborOf e2= armenia; store_all_correct[(turkey, neighborOf)]=armenia, azerbaijan, iran, greece, ...
            for line in self.train_data:
                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]
                if self.rwd:
                    self.store.append([e1,r,e2])
                self.store_all_correct[(e1, r)].add(e2)  #YM: there may exist multiple answers for the same query, i.e., same (e1,r) may mapping to different e2. store_all_correct will give all solution for the same query
            if self.rwd:
                ##self.store = np.array(self.store[:400])
                self.store = np.array(self.store)
                print(self.store.shape)
            ##self.queries=np.array(list(self.store_all_correct.keys())[:400], int)
            self.queries=np.array(list(self.store_all_correct.keys()), int)
            print(self.queries.shape)
        else:
            if self.mode == 'test':
                dataset = self.test_data
            if self.mode == 'dev':
                dataset = self.dev_data
            #same as above but on the test or dev dataset
            self.query_answers = defaultdict(set)
            for line in dataset:
                e1 = line[0]
                r = line[1]
                e2 = line[2]
                if e1 in self.entity_vocab and e2 in self.entity_vocab:
                    e1 = self.entity_vocab[e1]
                    r = self.relation_vocab[r]
                    e2 = self.entity_vocab[e2]
                    if self.rwd:
                        self.store.append([e1,r,e2])
                    self.query_answers[(e1,r)].add(e2)
            if self.rwd:
                self.store = np.array(self.store)
            #stores all the correct answers for a query in a graph in a dictionary over the full graph. This just includes everything, not just training data, hence the different code
            for line in self.full_graph:
                e1 = line[0]
                r = line[1]
                e2 = line[2]
                if e1 in self.entity_vocab and e2 in self.entity_vocab:
                    e1 = self.entity_vocab[e1]
                    r = self.relation_vocab[r]
                    e2 = self.entity_vocab[e2]
                    self.store_all_correct[(e1, r)].add(e2)
            ###self.queries=np.array(list(self.query_answers.keys()), int)
            ####
        
                    
    #UNDERSTOOD
    def yield_next_batch_train(self, labeller, rl):
        epoch = 1
        current_idx = 0
        if self.rwd:
            remaining = self.store.shape[0]
            random.shuffle(self.store)
        else:
            remaining = self.queries.shape[0]
            random.shuffle(self.queries)
        # shuffle list before beginning training
        while True:
            # reset for new epoch
            if remaining == 0:
                if self.rwd:
                    remaining = self.store.shape[0]
                    random.shuffle(self.store)
                else:
                    remaining = self.queries.shape[0]
                    random.shuffle(self.queries)
                epoch += 1
                current_idx = 0

            # get batch indices
            if remaining - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining -= self.batch_size
            else:
                if self.rwd:
                    batch_idx = np.arange(current_idx, self.store.shape[0])
                else:
                    batch_idx = np.arange(current_idx, self.queries.shape[0])
                remaining = 0

            # get batch
            if self.rwd:
                rl = False
                batch = self.store[batch_idx, :]
            else:
                batch = self.queries[batch_idx, :]
            
            # if doing supervised learning, randomly generates a list of indices of facts in the training data the length of which is the batch size
            # creates numpy array of the facts at those indexes
            if not self.rwd and not rl:
                #generates correct paths
                labels=[[],[],[]]
                for i in range(len(batch)):
                    correct = labeller.correct_path(batch[i])
                    #handle rollouts 
                    for i in range(self.num_rollouts):
                        labels[0].append(correct[0])
                        labels[1].append(correct[1])
                        labels[2].append(correct[2])

                #get indices where no path was found and delete them from the batch
                indices=np.argwhere(labels==np.array([-1,-1]))
                np.delete(labels,indices)
                np.delete(batch,indices)

            if rl:
                masked = []
                for i in range(len(batch)):
                    temp = labeller.get_random_rl_masking(batch[i])
                    masked.append(temp)

            #split these facts into their component parts
            e1 = batch[:,0]
            r = batch[:,1]
            if self.rwd:
                e2 = batch[:, 2]
            all_e2s = []
            #creates a list of all the right answers if (e1, r, ?) was the query. The reason for the for loop and not just assigning is making sure e1,r,e2,and all_e2s are in the same order
            if rl:
                for i in range(e1.shape[0]):
                    all_e2s.append(list(set(self.store_all_correct[(e1[i], r[i])]) - set(masked[i])))
            else:
                for i in range(e1.shape[0]):
                    all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            if self.rwd:
                assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            assert e1.shape[0] == r.shape[0] == len(all_e2s)
            if self.rwd:
                yield e1, r, e2, all_e2s, epoch
            else:
                if rl:
                    yield e1, r, all_e2s, masked, epoch
                else:
                    yield e1, r, all_e2s, labels, epoch

    #UNDERSTOOD
    # doesnt check rwd since we always test with original reward
    def yield_next_batch_test(self, labeller):
        remaining_triples = self.store.shape[0]
        ###remaining_queries = self.queries.shape[0]
        current_idx = 0
        while True:
            #return if we're out of queries
            ###if remaining_queries == 0:
            if remaining_triples == 0:
                return
            #move forwards 1 batch if we have more remaining queries than the batch size
            if remaining_triples - self.batch_size > 0:
            ###if remaining_queries - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
                ###remaining_queries -= self.batch_size
            #use the remaining queries since we have fewer than 1 batch left
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])
                ###batch_idx = np.arange(current_idx, self.queries.shape[0])
                remaining_triples = 0
                ###remaining_queries = 0
            #get the facts that batch_idx points to as well as the potential right answers for the (e1, r, ?) query
            batch = self.store[batch_idx, :]
            ###batch = self.queries[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            ###assert e1.shape[0] == r.shape[0] == len(all_e2s)
            #return a list of facts and and query answers where the query is e1 and r of an arbitrary fact
            yield e1, r, e2, all_e2s
            ###yield e1, r, all_e2s
