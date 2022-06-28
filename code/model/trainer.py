from __future__ import absolute_import
from __future__ import division
from matplotlib import pyplot as plt
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
import resource
import sys
from code.model.baseline import ReactiveBaseline
#from scipy.misc import logsumexp as lse
from scipy.special import logsumexp as lse
from code.data.vocab_gen import Vocab_Gen
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Trainer(object):
    def __init__(self, params, train_type, reward_type):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

        self.rl = train_type == "reinforcement"
        self.original_reward = reward_type == "original"

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = env(params, 'train', self.rl, self.original_reward)
        self.dev_test_environment = env(params, 'dev', True, True)
        self.test_test_environment = env(params, 'test', True, True)
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
        self.cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        # if RL, we collect the queries the agent didn't perform on
        if self.rl:
            self.needs_work_queries = []
            self.needs_work_scores = {}

        if not self.rl: # otherwise total iterations is fine as-is
            self.total_iterations = self.total_iterations_sl
            # self.total_iterations = self.total_epochs_sl * int(self.train_environment.batcher.train_set_length/self.batch_size)

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

    # the scores actually aren't exactly between 0 and 1, so we normalize them to that range to get a proper CCE error
    def normalize_scores(self, scores):
        scores = tf.cast(scores, dtype=tf.float32)
        scores = tf.divide(tf.subtract(scores, tf.reduce_min(scores)), tf.subtract(tf.reduce_max(scores), tf.reduce_min(scores)))
        return scores

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy

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

    def train(self, xdata, ydata_accuracy, ydata_loss):
        train_loss = 0.0
        self.batch_counter = 0
        self.first_state_of_test = False
        self.range_arr = np.arange(self.batch_size*self.num_rollouts)
        
        for episode in self.train_environment.get_episodes():

            self.batch_counter += 1
            model_state = self.agent.state_init
            prev_relation = self.agent.relation_init            

            # h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            query_relation = episode.get_query_relation()
            query_embedding = self.agent.get_query_embedding(query_relation)
            # get initial state
            state = episode.get_state()
            # for use with SL
            last_step = ["N/A"]*(self.batch_size*self.num_rollouts)
            # for each time step
            with tf.GradientTape() as tape:
                supervised_learning_loss = []
                loss_before_regularization = []
                logits_all = []

                for i in range(self.path_length):
                    loss, model_state, logits, idx, prev_relation, scores = self.agent.step(state['next_relations'],
                                                                                  state['next_entities'],
                                                                                  model_state, prev_relation, query_embedding,
                                                                                  state['current_entities'],  
                                                                                  range_arr=self.range_arr,
                                                                                  first_step_of_test = self.first_state_of_test)
                    if self.rl:
                        loss_before_regularization.append(loss)
                        logits_all.append(logits)
                    else:
                        active_length=scores.shape[0]
                        choices=scores.shape[1]

                        correct=np.full((active_length,choices),0)
                        for batch_num in range(len(episode.correct_path[i])):
                            try:
                                valid = episode.correct_path[i][batch_num][last_step[batch_num]]
                            except:
                                valid = episode.backtrack(batch_num)
                            correct[np.array([batch_num]*len(valid), int),np.array(valid, int)]=np.ones(len(valid))
                            #verify that the valid actions are encoded in the correct label correctly
                            if not sorted(list(set([int(x) for x in valid]))) == list(np.nonzero(correct[batch_num,:]==1)[0]) and not -1 in sorted(list(set([int(x) for x in valid]))):
                                print("ALERT")
                                print(sorted(list(set([int(x) for x in valid]))))
                                print(list(np.nonzero(correct[batch_num,:]==1)[0]))
                        last_step = idx.numpy()
                        loss = self.cce(tf.convert_to_tensor(correct), self.normalize_scores(scores))
                        
                        supervised_learning_loss.append(loss)
                    # action = np.squeeze(action, axis=1)  # [B,]
                    state = episode(idx)

                # get the final reward from the environment
                rewards = episode.get_reward()
                accuracy = np.sum((np.sum(np.reshape(rewards, (self.batch_size, self.num_rollouts)), axis=1) > 0))/self.batch_size
                xdata.append(float(max(xdata) + 1))
                ydata_accuracy.append(float(accuracy))

                if self.rl:
                    # update the list of incorrect queries. We have a dictionary with the number of times the agent got the query wrong
                    # and a list of the actual queries sorted by the values stored in that dictionary. That way, the queries the agent
                    # has gotten wrong the most stay at the top
                    incorrect_queries = [[e1, r, e2] for (e1, r, e2, reward) in zip(episode.start_entities, episode.query_relation, episode.end_entities, rewards) if reward == episode.negative_reward]
                    for query in incorrect_queries:
                        key = "_".join([str(x) for x in query])
                        if not key in list(self.needs_work_scores.keys()):
                            self.needs_work_scores[key] = 1
                            self.needs_work_queries.append(query)
                        else:
                            self.needs_work_scores[key] += 1
                    self.needs_work_queries.sort(key=lambda query: self.needs_work_scores["_".join([str(x) for x in query])])

                if self.rl:
                    # computed cumulative discounted reward
                    cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]
                    batch_total_loss = self.calc_reinforce_loss(cum_discounted_reward,loss_before_regularization,logits_all)
                    ydata_loss.append(float(batch_total_loss.numpy()))
                else:
                    sl_loss_float64 = [tf.cast(x, tf.float64) for x in supervised_learning_loss]
                    reduced_sum = tf.reduce_sum(sl_loss_float64,0)
                    square = tf.math.square(reduced_sum)
                    supervised_learning_total_loss =  tf.math.reduce_mean(square)
                    ydata_loss.append(float(supervised_learning_total_loss.numpy()))

            if self.rl:
                gradients = tape.gradient(batch_total_loss, self.agent.trainable_variables)
            else:
                gradients = tape.gradient(supervised_learning_total_loss, self.agent.trainable_variables)

            # print(len(self.agent.trainable_variables),self.agent.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))        

            # print statistics
            train_loss = 0.98 * train_loss + 0.02 * (batch_total_loss if self.rl else supervised_learning_total_loss)
            # train_loss1 = 0.98 * train_loss1 + 0.02 * loss1
            # print(batch_total_loss,loss1,train_loss,train_loss1)
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")
                
            logger.info("RL: {0:4d}, batch_counter: {1:4d}, num_hits: {2:7.4f}, avg. reward per batch {3:7.4f}, "
                        "num_ep_correct {4:4d}, avg_ep_correct {5:7.4f}, train loss {6:7.4f}".
                        format(int(self.rl),self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                                (num_ep_correct / self.batch_size),
                                train_loss))                
            # print('111111111111111111111111')
            #commented out to improve speed:
            # if self.batch_counter%self.eval_every == 0:
            #     with open(self.output_dir + '/scores.txt', 'a') as score_file:
            #         score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
            #     # os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
            #     # self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"
            #     self.test(beam=True, print_paths=False)

            # logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # gc.collect()
            if self.batch_counter >= self.total_iterations:
                return xdata, ydata_accuracy, ydata_loss

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

                loss, agent_mem, test_scores, test_action_idx, chosen_relation, _ = self.agent.step(state['next_relations'],
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

    #reset loss graph to add another set of data
    xdata = [float(0.0)]
    ydata_loss = [float(0.0)]
    ydata_accuracy = [float(0.0)]

    def train_rl(options, xdata, ydata_accuracy, ydata_loss, first):
        trainer = Trainer(options, "reinforcement", "original")
        if not first:
            trainer.agent.load_weights(options['model_dir'])
        xdata, ydata_accuracy, ydata_loss = trainer.train(xdata, ydata_accuracy, ydata_loss)
        trainer.agent.save_weights(options['model_dir'])
        return xdata, ydata_accuracy, ydata_loss, trainer.needs_work_queries[:len(trainer.needs_work_queries)//4]

    def train_sl(options, xdata, ydata_accuracy, ydata_loss, needs_work):
        options['dataset']['train'] = needs_work
        options['beta'] = options['beta_sl']
        options['Lambda'] = options['Lambda_sl']
        options['learning_rate'] = options['learning_rate_sl']
        options['random_masking_coef'] = 0
        options['total_epochs_sl'] = options['sl_start_checkpointing']
        trainer = Trainer(options, "supervised", "our")
        trainer.agent.load_weights(options['model_dir'])
        xdata, ydata_accuracy, ydata_loss = trainer.train(xdata, ydata_accuracy, ydata_loss)
        trainer.agent.save_weights(options['model_dir'])
        trainer.test()
        return xdata, ydata_accuracy, ydata_loss

    def make_sl_checkpoint(last_epoch, options, xdata, ydata_accuracy, ydata_loss):
        original_model_dir = options['model_dir']
        # make checkpoint folder
        options['output_dir'] += '/checkpoint_sl_epoch_'+str(last_epoch)
        os.mkdir(options['output_dir'])

        # make output folder
        os.mkdir(options['output_dir']+'/path_info/')
        options['path_logger_file'] = options['output_dir']+'/path_info/path_logger'
        options['log_file_name'] = options['output_dir'] +'/log.txt'

        # make model folder
        os.mkdir(options['output_dir']+'/model_weights/')
        options['model_dir'] = options['output_dir']+'/model_weights/'

        # make trainer
        trainer = Trainer(options, "reinforcement", "original")
        trainer.agent.load_weights(original_model_dir)

        # do RL training
        xdata, ydata_accuracy, ydata_loss = trainer.train(xdata, ydata_accuracy, ydata_loss)

        # create graph
        figure, axes = plt.subplots(1,2)
        loss_graph=axes[0]
        accuracy_graph=axes[1]
        loss_graph.set_xlim(float(1.0), max(xdata))
        accuracy_graph.set_xlim(float(1.0), max(xdata))
        loss_graph.set_ylim(min(float(0.0), min(ydata_loss)), max(ydata_loss))
        accuracy_graph.set_ylim(min(float(0.0), min(ydata_accuracy)), max(ydata_accuracy))
        line_loss, = loss_graph.plot(xdata, ydata_loss, 'r-', label="Loss")
        line_accuracy, = accuracy_graph.plot(xdata, ydata_accuracy, 'r-', label="Accuracy")
        line_loss.set_xdata(xdata)
        line_accuracy.set_xdata(xdata)
        line_loss.set_ydata(ydata_loss)
        line_accuracy.set_ydata(ydata_accuracy)
        plt.draw()
        plt.savefig(options['output_dir']+'/'+options['model_name']+".png")
        plt.close(figure)

        # do testing
        trainer.test()
        
        # save model
        trainer.agent.save_weights(options['model_dir'] + options['model_name'])

    def hp_tune(b, l, lr):
        xdata = [float(0.0)]
        ydata_loss = [float(0.0)]
        ydata_accuracy = [float(0.0)]
        # create SL Trainer
        options['beta'] = options['beta_sl']
        options['Lambda'] = options['Lambda_sl']
        options['learning_rate'] = options['learning_rate_sl']
        options['random_masking_coef'] = 0
        options['total_epochs_sl'] = options['sl_start_checkpointing']
        trainer = Trainer(options, "supervised", "our")
        xdata, ydata_accuracy, ydata_loss = trainer.train(xdata, ydata_accuracy, ydata_loss)
        # create graph
        figure, axes = plt.subplots(1,2)
        loss_graph=axes[0]
        accuracy_graph=axes[1]
        loss_graph.set_xlim(float(1.0), max(xdata))
        accuracy_graph.set_xlim(float(1.0), max(xdata))
        loss_graph.set_ylim(min(float(0.0), min(ydata_loss)), max(ydata_loss))
        accuracy_graph.set_ylim(min(float(0.0), min(ydata_accuracy)), max(ydata_accuracy))
        line_loss, = loss_graph.plot(xdata, ydata_loss, 'r-', label="Loss")
        line_accuracy, = accuracy_graph.plot(xdata, ydata_accuracy, 'r-', label="Accuracy")
        line_loss.set_xdata(xdata)
        line_accuracy.set_xdata(xdata)
        line_loss.set_ydata(ydata_loss)
        line_accuracy.set_ydata(ydata_accuracy)
        plt.draw()
        plt.savefig(options['output_dir']+'/'+"lr_"+str(lr)+"_b_"+str(b)+"_L_"+str(l)+".png")
        plt.close(figure)

    original_options = options.copy()
    first = True
    for x in range(options['num_cycles']):
        xdata, ydata_accuracy, ydata_loss, needs_work = train_rl(original_options.copy(), xdata, ydata_accuracy, ydata_loss, first)
        first = False
        xdata, ydata_accuracy, ydata_loss = train_sl(original_options.copy(), xdata, ydata_accuracy, ydata_loss, needs_work)

    ydata_accuracy = moving_average(ydata_accuracy, n=50)
    ydata_loss = moving_average(ydata_loss, n=50)
    xdata = np.array(list(range(ydata_accuracy.shape[0])))
    figure, axes = plt.subplots(1,2)
    loss_graph=axes[0]
    accuracy_graph=axes[1]
    loss_graph.set_xlim(float(1.0), max(xdata))
    accuracy_graph.set_xlim(float(1.0), max(xdata))
    loss_graph.set_ylim(min(float(0.0), min(ydata_loss)), max(ydata_loss))
    accuracy_graph.set_ylim(min(float(0.0), min(ydata_accuracy)), max(ydata_accuracy))
    line_loss, = loss_graph.plot(xdata, ydata_loss, 'r-', label="Loss")
    line_accuracy, = accuracy_graph.plot(xdata, ydata_accuracy, 'r-', label="Accuracy")
    line_loss.set_xdata(xdata)
    line_accuracy.set_xdata(xdata)
    line_loss.set_ydata(ydata_loss)
    line_accuracy.set_ydata(ydata_accuracy)
    plt.draw()
    plt.savefig(options['output_dir']+'/'+options['model_name']+".png")
    plt.close(figure)

    #beta
    # hp_tune(2, 0.02, 1e-3)
    # hp_tune(0.2, 0.02, 1e-3)
    # hp_tune(0.02, 0.02, 1e-3)
    # hp_tune(0.002, 0.02, 1e-3)
    # hp_tune(0.0002, 0.02, 1e-3)
    # hp_tune(0.02, 2, 1e-3)
    # hp_tune(0.02, 0.2, 1e-3)
    # hp_tune(0.02, 0.002, 1e-3)
    # hp_tune(0.02, 0.0002, 1e-3)

    # create SL Trainer
    # options['beta'] = options['beta_sl']
    # options['Lambda'] = options['Lambda_sl']
    # options['learning_rate'] = options['learning_rate_sl']
    # options['random_masking_coef'] = 0
    # options['total_epochs_sl'] = options['sl_start_checkpointing']
    # trainer = Trainer(options, "supervised", "our")

    # # Create checkpoint for pure RL run
    # last_epoch = 0
    # trainer.agent.save_weights(options['model_dir'])
    # make_sl_checkpoint(last_epoch, original_options.copy(), xdata.copy(), ydata_accuracy.copy(), ydata_loss.copy())
    # trainer.agent.load_weights(options['model_dir'])

    # # Create initial SL checkpoint
    # xdata, ydata_accuracy, ydata_loss = trainer.train(xdata, ydata_accuracy, ydata_loss)
    # last_epoch = trainer.total_epochs_sl

    # # Create first post-SL checkpoint
    # trainer.agent.save_weights(options['model_dir'])
    # make_sl_checkpoint(last_epoch, original_options.copy(), xdata.copy(), ydata_accuracy.copy(), ydata_loss.copy())
    # trainer.agent.load_weights(options['model_dir'])

    # # Create subsequent SL checkpoints
    # trainer.total_epochs_sl = options['sl_checkpoint_interval']

    # for ckpt in range(options['sl_checkpoints']):
    #     xdata, ydata_accuracy, ydata_loss = trainer.train(xdata, ydata_accuracy, ydata_loss)
    #     last_epoch += trainer.total_epochs_sl

    #     trainer.agent.save_weights(options['model_dir'])
    #     make_sl_checkpoint(last_epoch, original_options.copy(), xdata.copy(), ydata_accuracy.copy(), ydata_loss.copy())
    #     trainer.agent.load_weights(options['model_dir'])