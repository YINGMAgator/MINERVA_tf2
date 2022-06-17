from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
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
#import resource
import sys
from code.model.baseline import ReactiveBaseline
#from scipy.misc import logsumexp as lse
from scipy.special import logsumexp as lse
import dill
from code.data.vocab_gen import Vocab_Gen
import matplotlib
import matplotlib.pyplot as plt
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

matplotlib.use('Agg') 
class Trainer(object):
    def __init__(self, params, agent = None):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)
        self.rwd = self.train_rwd
        self.test = self.test_round
        if self.test:
            self.rwd = True
            ###self.rwd = False
        # allow passing an agent to the trainer for testing
        if agent !=None:
            self.agent = agent
        else:
            self.agent = Agent(params)
        # I don't want to load a 10gb file into memory 3 times so I'm just not creating these environemnts during this part of testing
        self.dev_test_environment = None #env(params, 'dev')
        self.test_environment = self.dev_test_environment
        if self.test:
            ###self.test_environment = env(params, False, 'test')
            self.test_environment = env(params, True, 'test')
            self.train_environment = None
            self.rev_relation_vocab = self.test_environment.grapher.rev_relation_vocab
            self.rev_entity_vocab = self.test_environment.grapher.rev_entity_vocab
        else:
            self.test_environment = None
            self.train_environment = env(params, self.rwd, 'train')
            self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
            self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        self.global_step = 0
        self.path_data = []

    def set_hpdependent(self, options):
        self.Lambda = options['Lambda']
        self.beta = options['beta']
        self.learning_rate = options['learning_rate']
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

    # the scores actually aren't exactly between 0 and 1, so we normalize them to that range to get a proper CCE error
    def normalize_scores(self, scores):
        scores = tf.cast(scores, dtype=tf.float32)
        scores = tf.divide(tf.subtract(scores, tf.reduce_min(scores)), tf.subtract(tf.reduce_max(scores), tf.reduce_min(scores)))
        return scores

    def train(self, use_RL, xdata, ydata_accuracy, ydata_loss, last_epoch):
        print("Beginning Training")
        self.first_state_of_test = False
        self.range_arr = np.arange(self.batch_size*self.num_rollouts)
        # cross entropy that we will use in our supervised learning implementation
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        average_accuracy = []
        average_loss = []
        for z, episode in enumerate(self.train_environment.get_episodes(use_RL)):
            graph_epoch = episode.epoch + last_epoch

            if (use_RL and episode.epoch > self.total_epochs_rl) or (not use_RL and episode.epoch > self.total_epochs_sl):
                return xdata, ydata_accuracy, ydata_loss, graph_epoch - 1

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
                    loss, model_state, logits, idx, prev_relation, scores = self.agent([state['next_relations'],
                                                                                  state['next_entities'],
                                                                                  model_state, prev_relation, query_embedding,
                                                                                  state['current_entities'],  
                                                                                  self.range_arr,
                                                                                  self.first_state_of_test])
                    #Step 2:
                    #some code to calculate loss between loss and prediction
                    if use_RL:
                        loss_before_regularization.append(loss)
                        logits_all.append(logits)
                    else: #use supervised learning
                        active_length=scores.shape[0]
                        choices=scores.shape[1]

                        correct=np.full((active_length,choices),0)
                        actions_test=np.array([], int)
                        for batch_num in range(len(episode.correct_path[i])):
                            try:
                                valid = episode.correct_path[i][batch_num][last_step[batch_num]]
                            except:
                                valid = episode.backtrack(batch_num)
                            # account for the dumb way I accounted for dead ends before
                            actions_test = np.concatenate((actions_test, [valid[0] if len(valid)!=0 else 0]))
                            correct[np.array([batch_num]*len(valid), int),np.array(valid, int)]=np.ones(len(valid))
                            #verify that the valid actions are encoded in the correct label correctly
                            if not sorted(list(set([int(x) for x in valid]))) == list(np.nonzero(correct[batch_num,:]==1)[0]) and not -1 in sorted(list(set([int(x) for x in valid]))):
                                print("ALERT")
                                print(sorted(list(set([int(x) for x in valid]))))
                                print(list(np.nonzero(correct[batch_num,:]==1)[0]))
                        last_step = idx.numpy() #actions_test if verifying labels
                        supervised_learning_loss.append(cce(tf.convert_to_tensor(correct),self.normalize_scores(scores)))
                        actions_test=actions_test.astype(int)
                    
                    state = episode(idx) #actions_test if verifying labels

                # batch num # of rows, each holding a path
                reward = episode.get_reward(self.rwd)

                #calculating the accuracy, or the portion of batches where the correct answer was found
                accuracy = np.sum((np.sum(np.reshape(reward, (self.batch_size, self.num_rollouts)), axis=1) > 0))/self.batch_size
                print("Accuracy "+ str(accuracy))
                
                # get the final reward from the environment and update the limits of the graphs accordingly
                # plus add the new data to the dataset
                if use_RL:
                    rewards = episode.get_reward(self.rwd)
                    # computed cumulative discounted reward
                    cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]
                    batch_total_loss = self.calc_reinforce_loss(cum_discounted_reward,loss_before_regularization,logits_all)
                    print("Reinforcement Learning Total Loss:")
                    print(batch_total_loss)

                    #append new data
                    average_loss.append(batch_total_loss)

                else: #use supervised learning
                    supervised_learning_total_loss =  tf.math.reduce_mean(tf.math.square(tf.reduce_sum(supervised_learning_loss,0)))
                    print("Supervised Learning Total Loss:")
                    print(supervised_learning_total_loss)

                    #append new data
                    average_loss.append(supervised_learning_total_loss)

                average_accuracy.append(accuracy)

                # update graph at new epoch
                if not graph_epoch in xdata:
                    xdata.append(graph_epoch)

                    # calculate averages
                    average_accuracy = np.array(average_accuracy)
                    average_loss = np.array(average_loss)

                    mean_acc = np.mean(average_accuracy)
                    mean_loss = np.mean(average_loss)

                    # reset lists
                    average_accuracy = []
                    average_loss = []

                    # add to dataset
                    ydata_accuracy.append(mean_acc)
                    ydata_loss.append(mean_loss)
                    
                print("Episode: "+str(z))
                print("Epoch: "+ str(episode.epoch))

            if use_RL:
                gradients = tape.gradient(batch_total_loss, self.agent.trainable_variables)
            else: #use supervised learning
                gradients = tape.gradient(supervised_learning_total_loss, self.agent.trainable_variables)

            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(gradients, self.agent.trainable_variables))        

    def testing(self, beam=False, print_paths=True, save_model = False, auc = False):
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
                loss, agent_mem, test_scores, test_action_idx, chosen_relation, scores = self.agent([state['next_relations'],
                                                                              state['next_entities'],
                                                                              model_state, previous_relation, query_embedding,
                                                                              state['current_entities'],  
                                                                              self.range_arr_test,
                                                                              self.first_state_of_test])
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
            rewards = episode.get_reward(True)  # [B*test_rollouts]
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
                    qr = self.test_environment.grapher.rev_relation_vocab[query_relation [b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    ###end_e = [self.rev_entity_vocab[e2] for e2 in episode.all_answers[b * self.test_rollouts]]
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

        print("len of paths and answers")
        print(len(list(paths.keys())))
        print(len(answers))
        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file + 'answers', 'w') as answer_file:
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

    #reset loss graph to add another set of data
    xdata = []
    ydata_loss = []
    ydata_accuracy = []

    def make_checkpoint(last_epoch, agent, options, xdata, ydata_accuracy, ydata_loss):
        # set rwd
        options['train_rwd'] = True

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
        trainer = Trainer(options, agent)
        
        # do RL training
        options['beta']=0.02
        options['Lambda']=0.02
        options['learning_rate']=1e-3
        trainer.set_hpdependent(options)
        xdata, ydata_accuracy, ydata_loss, last_epoch = trainer.train(True, xdata, ydata_accuracy, ydata_loss, last_epoch)
        
        # create graph
        figure, axes = plt.subplots(1,2)
        loss_graph=axes[0]
        accuracy_graph=axes[1]
        loss_graph.set_xlim(1, last_epoch)
        accuracy_graph.set_xlim(1, last_epoch)
        loss_graph.set_ylim(min(0, min(ydata_loss)), max(ydata_loss))
        accuracy_graph.set_ylim(min(0, min(ydata_accuracy)), max(ydata_accuracy))
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
        options['test_round'] = True
        tester = Trainer(options, trainer.agent)
        tester.testing()
        
        # save model
        trainer.agent.save_weights(options['model_dir'] + options['model_name'])

    # Training
    if not options['load_model']:
        trainer = Trainer(options)

        # Create checkpoint for no rl
        last_epoch = 0
        trainer.agent.save_weights(options['model_dir'])
        make_checkpoint(last_epoch, trainer.agent, options.copy(), xdata.copy(), ydata_accuracy.copy(), ydata_loss.copy())
        trainer.agent.load_weights(options['model_dir'])

        # Create initial SL checkpoint
        options['beta']=0.002
        options['Lambda']=2
        options['learning_rate']=0.001
        options['random_masking_coef']=0
        trainer.set_hpdependent(options)
        trainer.total_epochs_sl = options['sl_start_checkpointing']
        xdata, ydata_accuracy, ydata_loss, last_epoch = trainer.train(False, xdata, ydata_accuracy, ydata_loss, last_epoch)
        trainer.agent.save_weights(options['model_dir'])
        make_checkpoint(last_epoch, trainer.agent, options.copy(), xdata.copy(), ydata_accuracy.copy(), ydata_loss.copy())
        trainer.agent.load_weights(options['model_dir'])

        # Create subsequent SL checkpoints
        trainer.total_epochs_sl = options['sl_checkpoint_interval']
        for ckpt in range(options['sl_checkpoints']):
            xdata, ydata_accuracy, ydata_loss, last_epoch = trainer.train(False, xdata, ydata_accuracy, ydata_loss, last_epoch)
            trainer.agent.save_weights(options['model_dir'])
            make_checkpoint(last_epoch, trainer.agent, options.copy(), xdata.copy(), ydata_accuracy.copy(), ydata_loss.copy())
            trainer.agent.load_weights(options['model_dir'])
    else:
        options['test_round'] = True
        tester = Trainer(options)
        tester.agent.load_weights(options['saved_model_dir'])
        tester.testing()