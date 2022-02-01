from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
from pprint import pprint

#UNDERSTOOD
def read_options():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--LSTM_layers", default=1, type=int)
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--total_iterations", default=2000, type=int)

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
    # parser.add_argument("--nell_query", default='all', type=str)

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    parsed['input_files'] = [parsed['data_input_dir'] + '/' + parsed['input_file']]

    parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
    parsed['train_entity_embeddings'] = (parsed['train_entity_embeddings'] == 1)
    parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)

    parsed['pretrained_embeddings_action'] = ""
    parsed['pretrained_embeddings_entity'] = ""

    parsed['output_dir'] = parsed['base_output_dir'] + '/' + str(uuid.uuid4())[:4]+'_'+str(parsed['path_length'])+'_'+str(parsed['beta'])+'_'+str(parsed['test_rollouts'])+'_'+str(parsed['Lambda'])

    parsed['model_dir'] = parsed['output_dir']+'/'+ 'model/'

    parsed['load_model'] = (parsed['load_model'] == 1)

    ##Logger##
    parsed['path_logger_file'] = parsed['output_dir']
    parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'
    os.makedirs(parsed['output_dir'])
    os.mkdir(parsed['model_dir'])
    with open(parsed['output_dir']+'/config.txt', 'w') as out:
        pprint(parsed, stream=out)

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    return parsed
