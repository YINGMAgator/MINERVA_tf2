from __future__ import absolute_import
from __future__ import division
import enum
from torch import float32

from tqdm import tqdm
import time
import os
import csv
import logging
import numpy as np
import tensorflow as tf
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
import codecs
from collections import defaultdict
import gc
#import resource
import sys
from code.model.baseline import ReactiveBaseline
#from scipy.misc import logsumexp as lse
from scipy.special import logsumexp as lse
from code.data.vocab_gen import Vocab_Gen
import matplotlib.pyplot as plt
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params):
        #set up loss graph
        plt.show()
        self.axes=plt.gca()
        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.xdata = []
        self.ydata = []
        self.line, = self.axes.plot(self.xdata, self.ydata, 'r-')

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        self.global_step = 0
        self.decaying_beta = tf.keras.optimizers.schedules.ExponentialDecay(self.beta,decay_steps=200,decay_rate=0.90, staircase=True)
        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    #total loss encorporating 
    def calc_reinforce_loss(self,cum_discounted_reward,loss_all,logits_all):
        loss = tf.stack(loss_all, axis=1)  # [B, T]

        self.tf_baseline = self.baseline.get_baseline_value()
        # self.pp = tf.Print(self.tf_baseline)
        # multiply with rewards
        final_reward = cum_discounted_reward - self.tf_baseline
        # reward_std = tf.sqrt(tf.reduce_mean(tf.square(final_reward))) + 1e-5 # constant addded for numerical stability
        reward_mean, reward_var = tf.nn.moments(x=final_reward, axes=[0, 1])
        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.math.divide(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)  # [B, T]
        # self.loss_before_reg = loss
        # print(self.decaying_beta(self.global_step) )
        # loss1= tf.reduce_mean(loss) 
        total_loss = tf.reduce_mean(loss) - self.decaying_beta(self.global_step) * self.entropy_reg_loss(logits_all)  # scalar
        # total_loss = tf.reduce_mean(loss)  # scalar
        # print(self.decaying_beta(self.global_step))
        return total_loss

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        #we take the average because of the batch size being greater than 1
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy


    #the return we all know and love, discounted by gamma
    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = np.zeros([rewards.shape[0]])  # [B]
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])  # [B, T]
        cum_disc_reward[:,
        self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def train(self):
        # import pdb
        # pdb.set_trace()
        # fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        self.batch_counter = 0
        self.first_state_of_test = False
        self.range_arr = np.arange(self.batch_size*self.num_rollouts)

        #cross entropy that we will use in our supervised learning implementation
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
        for z, episode in enumerate(self.train_environment.get_episodes()):
            self.batch_counter += 1
            model_state = self.agent.state_init
            prev_relation = self.agent.relation_init            

            # h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            query_relation = episode.get_query_relation()
            query_embedding = self.agent.get_query_embedding(query_relation)
            # get initial state
            state = episode.get_state()
            
            with tf.GradientTape() as tape:
                supervised_learning_loss = []
                loss_before_regularization = []
                logits_all = []
                # for each time step
                for i in range(self.path_length):
                    #Step 1:
                    #temporarily replace the idx with the brute force answer. Essentially, we mute the
                    #agent and just verify that the brute force algo. is correct, otherwise we cannot use
                    #it as a metric
                    loss, model_state, logits, idx, prev_relation, scores = self.agent.step(state['next_relations'],
                                                                                  state['next_entities'],
                                                                                  model_state, prev_relation, query_embedding,
                                                                                  state['current_entities'],  
                                                                                  range_arr=self.range_arr,
                                                                                  first_step_of_test = self.first_state_of_test)
                    #Step 2:
                    #some code to calculate loss between loss and prediction
                    ###functional RL code###loss_before_regularization.append(loss)
                    ###functional RL code###logits_all.append(logits)
                    # action = np.squeeze(action, axis=1)  # [B,]

                    ##CODE FOR SUPERVISED LEARNING LOSS
                    #idx=episode.correct_path[i]
                    active_length=scores.shape[0]
                    correct=np.full((active_length,200),0)
                    correct[np.arange(0,active_length),episode.correct_path[i]]=np.ones(active_length)
                    supervised_learning_loss.append(cce(tf.convert_to_tensor(correct),scores))

                    #create a tensor of size (2560,200) where in each of the 2560 rows, the tensor of length 200 has -9.9999000e+04 at every index except the index specified in the correct path, which has a 1
                    #then i take the cross entropy loss between the scores outputted by the network and the correct answers
                    #get the sum of the 200 long vectors so you have a 2560,1 vector
                    #append this to a list so you can take the sum over the three steps for each batch and then take the average of the batches
                    ##END CODE FOR SUPERVISED LEARNING LOSS

                    #gets the correct step of the correct path from the object variable
                    state = episode(idx)
                # get the final reward from the environment
                ###functional RL code###rewards = episode.get_reward()
    
                # computed cumulative discounted reward
                ###functional RL code###cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]
    
                ###functional RL code###batch_total_loss = self.calc_reinforce_loss(cum_discounted_reward,loss_before_regularization,logits_all)
                supervised_learning_total_loss =  tf.math.reduce_mean(tf.math.square(tf.reduce_sum(supervised_learning_loss,0)))
                print("Supervised Learning Total Loss:")
                print(supervised_learning_total_loss)

                #increment x of graph
                self.axes.set_xlim(0,self.axes.get_xlim()[1]+1)
                #change y of graph if needed
                if supervised_learning_total_loss>self.axes.get_ylim()[1]:
                    self.axes.set_ylim(self.axes.get_ylim()[0],supervised_learning_total_loss)
                elif supervised_learning_total_loss<self.axes.get_ylim()[0]:
                    self.axes.set_ylim(supervised_learning_total_loss,self.axes.get_ylim()[1])
                #regen line
                self.line, = self.axes.plot(self.xdata, self.ydata, 'r-')
                #append new data
                self.xdata.append(z)
                self.ydata.append(supervised_learning_total_loss)
                #populate new line
                self.line.set_xdata(self.xdata)
                self.line.set_ydata(self.ydata)
                #draw everything and briefly wait
                plt.draw()
                plt.pause(1e-17)
                time.sleep(0.1)

            ###functional RL code###gradients = tape.gradient(batch_total_loss, self.agent.trainable_variables)
            gradients = tape.gradient(supervised_learning_total_loss, self.agent.trainable_variables)
            # print(len(self.agent.trainable_variables),self.agent.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))        

            
            # print statistics
            ###functional RL code###train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            # train_loss1 = 0.98 * train_loss1 + 0.02 * loss1
            # print(batch_total_loss,loss1,train_loss,train_loss1)
            ###functional RL code###avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            ###functional RL code###reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            ###functional RL code###reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            ###functional RL code###reward_reshape = (reward_reshape > 0)
            ###functional RL code###num_ep_correct = np.sum(reward_reshape)
            ###functional RL code###if np.isnan(train_loss):
            ###functional RL code###    raise ArithmeticError("Error in computing loss")
            ###functional RL code###    
            ###functional RL code###logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
            ###functional RL code###            "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
            ###functional RL code###            format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
            ###functional RL code###                    (num_ep_correct / self.batch_size),
            ###functional RL code###                    train_loss))                
            # print('111111111111111111111111')
            ###functional RL code###if self.batch_counter%self.eval_every == 0:
            ###functional RL code###    with open(self.output_dir + '/scores.txt', 'a') as score_file:
            ###functional RL code###        score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
            ###functional RL code###    os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
            ###functional RL code###    self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"
            ###functional RL code###    self.test(beam=True, print_paths=False)

            # logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, beam=False, print_paths=False, save_model = True, auc = False):
        batch_counter = 0
        paths = defaultdict(list)
        answers = []
        # feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1

            temp_batch_size = episode.no_examples
            
            query_relation = episode.get_query_relation()
            query_embedding = self.agent.get_query_embedding(query_relation)
            
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            # get initial state
            state = episode.get_state()
            
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            layer_state = tf.unstack(agent_mem, self.LSTM_layers)
            model_state = [tf.unstack(s, 2) for s in layer_state]            
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            self.range_arr_test = np.arange(temp_batch_size * self.test_rollouts)
            # feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)

            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            # for each time step
            for i in range(self.path_length):
                if i == 0:
                    self.first_state_of_test = True

                loss, agent_mem, test_scores, test_action_idx, chosen_relation = self.agent.step(state['next_relations'],
                                                                              state['next_entities'],
                                                                              model_state, previous_relation, query_embedding,
                                                                              state['current_entities'],  
                                                                              range_arr=self.range_arr_test,
                                                                              first_step_of_test = self.first_state_of_test)
                agent_mem = tf.stack(agent_mem)
                agent_mem = agent_mem.numpy()
                test_scores = test_scores.numpy()
                test_action_idx = test_action_idx.numpy()
                chosen_relation = chosen_relation.numpy()
                if beam:
                    k = self.test_rollouts
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx//self.max_num_actions
                    x = idx%self.max_num_actions

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y, :]

                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                previous_relation = chosen_relation
                layer_state = tf.unstack(agent_mem, self.LSTM_layers)
                model_state = [tf.unstack(s, 2) for s in layer_state]   
                ####logger code####
                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                ####################
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]
            if beam:
                self.log_probs = beam_probs

            ####Logger code####

            if print_paths:
                self.entity_trajectory.append(
                    state['current_entities'])


            # ask environment for final reward
            rewards = episode.get_reward()  # [B*test_rollouts]
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            AP = 0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        if reward_reshape[b,r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in  sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None


                if answer_pos != None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1
                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0/((answer_pos+1))
                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    for r in sorted_indx[b]:
                        indx = b * self.test_rollouts + r
                        if rewards[indx] == self.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(self.rev_entity_vocab[se[b,r]]+'\t'+ self.rev_entity_vocab[ce[b,r]]+'\t'+ str(self.log_probs[b,r])+'\n')
                        paths[str(qr)].append(
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
                                        self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' + str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')
                    paths[str(qr)].append("#####################\n")

            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples
        # if save_model:
        #     if all_final_reward_10 >= self.max_hits_at_10:
        #         self.max_hits_at_10 = all_final_reward_10
        #         self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir+'/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("auc: {0:7.4f}".format(auc))

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))

if __name__ == '__main__':

    # read command line options
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    #read dataset
        
    options['dataset']={}
    Dataset_list=['train','test','dev','graph']
    for dataset in Dataset_list:
        input_file = options['data_input_dir']+dataset+'.txt'
        ds = []
        with open(input_file) as raw_input_file:
            csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            for line in csv_file:
                ds.append(line)
        options['dataset'][dataset]=ds

    # input_file = options['data_input_dir']+'test.txt'
    # options['test_data'] = []
    # with open(input_file) as raw_input_file:
    #     csv_file = csv.reader(raw_input_file, delimiter = '\t' )
    #     for line in csv_file:
    #         options['test_data'].append(line)  

    # input_file = options['data_input_dir']+'dev.txt'
    # options['dev_data'] = []
    # with open(input_file) as raw_input_file:
    #     csv_file = csv.reader(raw_input_file, delimiter = '\t' )
    #     for line in csv_file:
    #         options['dev_data'].append(line)  
            
    # input_file = options['data_input_dir']+'graph.txt'
    # options['graph_data'] = []
    # with open(input_file) as raw_input_file:
    #     csv_file = csv.reader(raw_input_file, delimiter = '\t' )
    #     for line in csv_file:
    #         options['graph_data'].append(line)  
    ds = []
    input_file = options['data_input_dir']+'full_graph.txt'
    if os.path.isfile(input_file):
        with open(input_file) as raw_input_file:
            csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            for line in csv_file:
                ds.append(line)  
    else:
        for dataset in Dataset_list:
            ds = ds + options['dataset'][dataset]
    options['dataset']['full_graph'] = ds
    # data_read_in =temp_fullgraph.split('\n')
    # options['fullgraph_data'] = []
    # for line in data_read_in:
    #     options['fullgraph_data'].append(line.split('\t'))        
    
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    vocab=Vocab_Gen(Datasets=[options['dataset']['train'],options['dataset']['test'],options['dataset']['graph']])
    options['relation_vocab'] = vocab.relation_vocab
    
    options['entity_vocab'] = vocab.entity_vocab
    # print(len(options['entity_vocab'] ))
    # options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    
    # options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    print(len(options['entity_vocab'] ))
    logger.info('Reading mid to name map')
    mid_to_word = {}
    # with open('/iesl/canvas/rajarshi/data/RL-Path-RNN/FB15k-237/fb15k_names', 'r') as f:
    #     mid_to_word = json.load(f)
    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    save_path = ''
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = False
    # config.log_device_placement = False

    # print('this is the right code,uuuuuuuuuuuuuyyyyyyyyyyyyyyyy')
    #Training
    if not options['load_model']:
        trainer = Trainer(options)
    # with tf.compat.v1.Session(config=config) as sess:
        # sess.run(trainer.initialize())

        trainer.train()
        save_path = trainer.save_path
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir

        # tf.compat.v1.reset_default_graph()
    #Testing on test with best model
    # else:
    #     logger.info("Skipping training")
    #     logger.info("Loading model from {}".format(options["model_load_dir"]))

    # trainer = Trainer(options)
    # if options['load_model']:
    #     save_path = options['model_load_dir']
    #     path_logger_file = trainer.path_logger_file
    #     output_dir = trainer.output_dir
    # with tf.compat.v1.Session(config=config) as sess:
    #     trainer.initialize(restore=save_path, sess=sess)

    #     trainer.test_rollouts = 100

    #     os.mkdir(path_logger_file + "/" + "test_beam")
    #     trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
    #     with open(output_dir + '/scores.txt', 'a') as score_file:
    #         score_file.write("Test (beam) scores with best model from " + save_path + "\n")
    #     trainer.test_environment = trainer.test_test_environment
    #     trainer.test_environment.test_rollouts = 100

    #     trainer.test(sess, beam=True, print_paths=True, save_model=False)
