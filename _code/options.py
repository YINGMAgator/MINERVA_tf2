from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
from pprint import pprint
from tensorflow import keras

#UNDERSTOOD
def read_options():
    parser = argparse.ArgumentParser()
    #the following arguments are only for use during hyperparameter tuning and should be commented out when not in use
    parser.add_argument("--hp_type", default=0, type=str)
    parser.add_argument("--hp_level", default=0, type=str)
    #use a lr schedule instead of a fixed lr
    parser.add_argument("--schedule", default=0, type=str)
    parser.add_argument("--rate", default=0, type=str)
    #flag for whether we are just running the program to generate labels for a dataset
    parser.add_argument("--label_gen", default=0, type=int)
    #for the training data
    parser.add_argument("--data_input_dir", default="", type=str)
    parser.add_argument("--input_file", default="train.txt", type=str)
    #for partially structured NLP queries on the movies dataset??
    parser.add_argument("--create_vocab", default=0, type=int)
    parser.add_argument("--vocab_dir", default="", type=str)
    #max number of edges to consider at a node
    parser.add_argument("--max_num_actions", default=200, type=int)
    #1-3 for tasks S1-S3, where the maximum path length is 1, 2, and 3 respectively. Longer path lengths correspond to longer logical chains.
    parser.add_argument("--path_length", default=3, type=int)
    #number of nodes in hidden layer (not sure which network)
    parser.add_argument("--hidden_size", default=50, type=int)
    #this corresponds to variable d in the paper used when defining the size of the embedding matrices. I assume length carries a tradeoff of increased precision in exchange for decreased speed and higher mem. requirements. Maybe overfitting
    parser.add_argument("--embedding_size", default=50, type=int)
    #number of queries
    parser.add_argument("--batch_size", default=128, type=int)
    #set gradient values above a certain threshold to this to avoid the exploding gradient problem commonly associated with LSTMs
    parser.add_argument("--grad_clip_norm", default=5, type=int)
    #coefficient of the squared magnitude regularization term
    parser.add_argument("--l2_reg_const", default=1e-2, type=float)
    #controls weight update sizes
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    #controls the weight of the entropy regularization term
    parser.add_argument("--beta", default=1e-2, type=float)
    #reward = 1 if final query answer is correct, 0 otherwise
    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=0, type=float)
    #discounting for the return over the episode g=y*r+y^2*r2+y^3*r3...
    parser.add_argument("--gamma", default=1, type=float)
    #directory to output logs
    parser.add_argument("--log_dir", default="./logs/", type=str)
    parser.add_argument("--log_file_name", default="reward.txt", type=str)
    parser.add_argument("--output_file", default="", type=str)
    #number of rollouts to perform to simulate the second expectation in the function given on page 4
    parser.add_argument("--num_rollouts", default=20, type=int)
    parser.add_argument("--test_rollouts", default=100, type=int)
    #self explanatory
    parser.add_argument("--LSTM_layers", default=1       , type=int)
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--total_iterations", default=2000, type=int)
    parser.add_argument("--total_epochs_sl", default=2000, type=int)
    parser.add_argument("--total_epochs_rl", default=2000, type=int)
    #SL hyperparameters
    parser.add_argument("--beta_sl", default=0.02, type=float)
    parser.add_argument("--Lambda_sl", default=0.02, type=float)
    parser.add_argument("--learning_rate_sl", default=1e-3, type=float)
    
    #not sure about this one
    parser.add_argument("--Lambda", default=0.0, type=float)
    #max pooling for the PATH-BASELINE
    parser.add_argument("--pool", default="max", type=str)
    parser.add_argument("--eval_every", default=100, type=int)
    #can be switched off for the compared methods that use symbolic logic (i believe) instead of vectorized embeddings
    parser.add_argument("--use_entity_embeddings", default=0, type=int)
    parser.add_argument("--train_entity_embeddings", default=0, type=int)
    parser.add_argument("--train_relation_embeddings", default=1, type=int)
    parser.add_argument("--model_load_dir", default="", type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--nell_evaluation", default=0, type=int)
    # for controlling generating last results
    parser.add_argument("--train_rwd", default=0, type=int)
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--model_name", default="test", type=str)
    parser.add_argument("--test_round", default=0, type=int) # not specified at cmd line; changed by code later
    parser.add_argument("--sl", default=1, type=int)
    parser.add_argument("--save_model", default=1, type=int)
    parser.add_argument("--saved_model_dir", default="none", type=str)
    # deprecated; we aren't doing order RL->SL tests anymore
    # parser.add_argument("--order_swap", default=0, type=int)
    parser.add_argument("--sl_start_checkpointing", default=2, type=int)
    parser.add_argument("--sl_checkpoints", default=8, type=int)
    parser.add_argument("--sl_checkpoint_interval", default=1, type=int)
    # random masking parameter
    parser.add_argument("--random_masking_coef", default=0, type=float)
    # # special training conditions for fb60k
    parser.add_argument("--fb60k", default=0, type=int)
    # special option to tune hyperparameters for a dataset
    parser.add_argument("--tune_hp", default=0, type=int)

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # ints to bools
    parsed['train_rwd'] = parsed['train_rwd'] == 1
    parsed['test'] = parsed['test'] == 1
    parsed['test_round'] = parsed['test_round'] == 1
    parsed['sl'] = parsed['sl'] == 1
    parsed['save_model'] = parsed['save_model'] == 1
    # parsed['fb60k'] = parsed['fb60k'] == 1
    parsed['tune_hp'] = parsed['tune_hp'] == 1
    # parsed['order_swap'] = parsed['order_swap'] == 1 
    #dataset name
    parsed['dataset_name']=parsed['base_output_dir'][7:-1]

    parsed['input_files'] = [parsed['data_input_dir'] + '/' + parsed['input_file']]

    parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
    parsed['train_entity_embeddings'] = (parsed['train_entity_embeddings'] == 1)
    parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)

    parsed['pretrained_embeddings_action'] = ""
    parsed['pretrained_embeddings_entity'] = ""

    parsed['output_dir'] = parsed['base_output_dir'] + '/' + str(uuid.uuid4())[:4]+'_'+parsed['model_name']+"_"+str(parsed['total_epochs_sl'])+"_"+str(parsed['total_epochs_rl'])

    parsed['model_dir'] = parsed['output_dir']+'/'+ 'model/'

    parsed['load_model'] = (parsed['load_model'] == 1)

    #handle lr schedule
    if parsed['schedule']==1:
        schedule=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=parsed['learning_rate'],decay_steps=parsed['total_epochs'],decay_rate=parsed['rate'])
        parsed['learning_rate']=schedule

    ##Logger##
    parsed['path_logger_file'] = parsed['output_dir']+'/path_info/path_logger'
    parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'
    os.makedirs(parsed['output_dir'])
    os.mkdir(parsed['model_dir'])
    os.mkdir(parsed['output_dir']+'/path_info/')
    with open(parsed['output_dir']+'/config.txt', 'w') as out:
        pprint(parsed, stream=out)

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    return parsed
