from __future__ import absolute_import
from __future__ import division
from cProfile import label
from cmath import sqrt
import enum
from math import ceil
from cv2 import accumulate
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
import pickle
from code.data.vocab_gen import Vocab_Gen
import matplotlib.pyplot as plt
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params, environment):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

        #set up loss graph, including x range since we already know the number of training iterations that will occur
        plt.show()
        self.axes=plt.gca()
        self.axes.set_xlim(0, self.total_iterations)
        self.axes.set_ylim(0, 1) #since we don't want the y lim to get reset ever, even when we switch from SL to RL

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = environment
        #self.train_environment = env(params, 'train')
        # I don't want to load a 10gb file into memory 3 times so I'm just not creating these environemnts during this part of testing
        self.dev_test_environment = None #env(params, 'dev')
        self.test_test_environment = None #env(params, 'test')
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

    def get_test_actions(self, correct):
        used_indexes=[]
        actions=[]
        chosen =np.where(correct==1)
        for count, index in enumerate(chosen[0]):
            if index in used_indexes:
                continue
            used_indexes += [index]
            actions += [chosen[1][count]]
        return actions

    def train(self,use_RL):
        #reset loss graph to add another set of data
        self.xdata = []
        self.ydata = []
        self.line, = self.axes.plot(self.xdata, self.ydata, 'r-' if use_RL else 'b-', label=("reinforcement " if use_RL else "supervised ")+"learning loss")

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
            
            last_step = ["N/A"]*(self.batch_size*self.num_rollouts)
            
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
                    if use_RL:
                        loss_before_regularization.append(loss)
                        logits_all.append(logits)
                        # action = np.squeeze(action, axis=1)  # [B,]
                    else: #use supervised learning
                        active_length=scores.shape[0]
                        correct=np.full((active_length,200),0)

                        actions_test=np.array([], int)
                        for batch_num in range(len(episode.correct_path[i])):
                            try:
                                valid = episode.correct_path[i][batch_num][last_step[batch_num]]
                            except:
                                print("this should never happen")
                                valid = episode.backtrack(batch_num)
                            # valid = list(set(valid))
                            actions_test = np.concatenate((actions_test, [valid[0]]))
                            correct[np.array([batch_num]*len(valid), int),np.array(valid, int)]=np.ones(len(valid))
                        last_step = idx.numpy()
                        supervised_learning_loss.append(cce(tf.convert_to_tensor(correct),scores))
                        actions_test=actions_test.astype(int)
                        # #we fill 2 arrays up, one with valid actions from where the agent is at and another with the batch that action
                        # #is relevant in
                        # indices=np.array([], int)
                        # actions=np.array([], int)

                        # for batch_num in range(len(episode.correct_path[i])):
                        #     #problem that if agent takes a step such that they can't get to a reward, their last step won't be a key in the dictionary
                        #     #find the action that goes back to the previous state
                        #     #solution: allow the agent to just learn nothing extra from these cases after they already happen. Instead, they learn from the fact that
                        #     # in the correct answer on the previous step, the action they took was marked as 0 because they shouldn't take it. Thus, the agent is already learning
                        #     # not to take that action
                        #     try:
                        #         valid = episode.correct_path[i][batch_num][last_step[batch_num]]
                        #     except:
                        #         valid = episode.backtrack(batch_num)
                        #     actions = np.concatenate((actions, valid))
                        #     indices = np.concatenate((indices, [batch_num] * len(valid)))
                        # #create the correct matrix pre-rollout, then roll it out so all the batches are effectively covered
                        # indices = indices.astype(int)
                        # actions = actions.astype(int)
                        # active_length=scores.shape[0]
                        # correct=np.full((active_length,200),0)
                        # correct[indices, actions] = np.ones(len(indices))
                        # ##MAKING SURE LABELS WORK
                        # actions = self.get_test_actions(correct)
                        # # ##fin
                        # # correct = np.repeat(correct, self.num_rollouts, axis=0)
                        # correct = correct.astype(int)
                        # supervised_learning_loss.append(cce(tf.convert_to_tensor(correct),scores))

                        # indices = np.repeat(indices, self.num_rollouts)
                        # actions = np.repeat(actions, self.num_rollouts)
                        # indices=indices.astype(int)
                        # actions=actions.astype(int)
                        # active_length=scores.shape[0]
                        # correct=np.full((active_length,200),0)
                        # correct[indices, actions] = np.ones(len(indices))
                        # print(np.count_nonzero(correct[len(episode.correct_path[i])+1:,:]==1))
                        # supervised_learning_loss.append(cce(tf.convert_to_tensor(correct),scores))

                        # update last step for all batches
                        #last_step = idx.numpy()
                        #last_step = actions

                    # code for testing if our label gen method is valid
                    # actions_test=np.array([], int)
                    # for batch_num in range(len(episode.correct_path[i])):
                    #     valid = episode.correct_path[i][batch_num][last_step[batch_num]]
                    #     actions_test = np.concatenate((actions_test, [valid[0]]))
                    # last_step=actions_test
                    # #actions_test = np.repeat(actions_test, self.num_rollouts)
                    # actions_test = actions_test.astype(int)
                    #gets gets the new state from the action chosen by the agent
                    #state = episode(idx)
                    state = episode(idx)

                #calculating the accuracy, or the portion of batches where the correct answer was found
                accuracy = np.sum((np.sum(np.reshape(episode.get_reward(), (self.batch_size, self.num_rollouts)), axis=1) > 0))/self.batch_size
                print("Accuracy "+ str(accuracy))
                
                # get the final reward from the environment and update the limits of the graphs accordingly
                # plus add the new data to the dataset
                if use_RL:
                    rewards = episode.get_reward()
                    # computed cumulative discounted reward
                    cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]
                    batch_total_loss = self.calc_reinforce_loss(cum_discounted_reward,loss_before_regularization,logits_all)
                    print("Reinforcement Learning Total Loss:")
                    print(batch_total_loss)

                    #change y of graph if needed
                    if batch_total_loss>self.axes.get_ylim()[1]:
                        self.axes.set_ylim(self.axes.get_ylim()[0],batch_total_loss)
                    elif batch_total_loss<self.axes.get_ylim()[0]:
                        self.axes.set_ylim(batch_total_loss,self.axes.get_ylim()[1])

                    #append new data
                    self.xdata.append(z)
                    self.ydata.append(batch_total_loss)

                else: #use supervised learning
                    supervised_learning_total_loss =  tf.math.reduce_mean(tf.math.square(tf.reduce_sum(supervised_learning_loss,0)))
                    print("Supervised Learning Total Loss:")
                    print(supervised_learning_total_loss)

                    #change y of graph if needed
                    if supervised_learning_total_loss>self.axes.get_ylim()[1]:
                        self.axes.set_ylim(self.axes.get_ylim()[0],supervised_learning_total_loss)
                    elif supervised_learning_total_loss<self.axes.get_ylim()[0]:
                        self.axes.set_ylim(supervised_learning_total_loss,self.axes.get_ylim()[1])

                    #append new data
                    self.xdata.append(z)
                    self.ydata.append(supervised_learning_total_loss)
                
                #regen line
                self.line, = self.axes.plot(self.xdata, self.ydata, 'r-' if use_RL else 'b-', label=("reinforcement " if use_RL else "supervised ")+"learning loss")
                #populate new line
                self.line.set_xdata(self.xdata)
                self.line.set_ydata(self.ydata)
                #draw everything and briefly wait
                plt.draw()
                #commented out because I need this to run in the background so I can work on other stuff
                #plt.pause(1e-17)
                time.sleep(0.1)

            if use_RL:
                gradients = tape.gradient(batch_total_loss, self.agent.trainable_variables)
            else: #use supervised learning
                gradients = tape.gradient(supervised_learning_total_loss, self.agent.trainable_variables)
            # print(len(self.agent.trainable_variables),self.agent.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))        

            if self.batch_counter >= self.total_iterations:
                plt.savefig("C:\\Users\\owenb\\OneDrive\\Documents\\GitHub\\MINERVA_tf2\\hyperparameter testing results\\FB15K\\"+self.hp_type+"\\"+self.hp_level+"_advanced_labels.png")
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

#if __name__ == '__main__':
def setup():
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
    
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    vocab=Vocab_Gen(Datasets=[options['dataset']['train'],options['dataset']['test'],options['dataset']['graph']])
    options['relation_vocab'] = vocab.relation_vocab
    
    options['entity_vocab'] = vocab.entity_vocab
    
    print(len(options['entity_vocab'] ))
    logger.info('Reading mid to name map')
    mid_to_word = {}

    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    save_path = ''

    return options, env(options, 'train')
    #return Trainer(options)
    # #Training
    # if not options['load_model']:
    #     trainer = Trainer(options)

    #     #one training pass supervised learning, one training pass reinforcement learning
    #     print("training with supervised learning")
    #     trainer.train(False)
    #     #commented out because we will only be doing hyperparameter tuning on SL
    #     #print("training with reinforcement learning")
    #     #trainer.train(True)
    #     save_path = trainer.save_path
    #     path_logger_file = trainer.path_logger_file
    #     output_dir = trainer.output_dir

def train(options, env):
    trainer = Trainer(options, env)
    print("training with supervised learning")
    trainer.train(False)
    #commented out because we will only be doing hyperparameter tuning on SL
    #print("training with reinforcement learning")
    #trainer.train(True)
    save_path = trainer.save_path
    path_logger_file = trainer.path_logger_file
    output_dir = trainer.output_dir