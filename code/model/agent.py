import numpy as np
import tensorflow as tf


class Agent(tf.keras.Model):

    def __init__(self, params):
        super(Agent, self).__init__()
        #attach all of the parameters passed to the trainer to the agent
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)

        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        #incorporates the rollouts into the batch size somehow
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2 #lower because we aren't storing entity encodings
        self.dense1 = tf.keras.layers.Dense(4 * self.hidden_size, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(self.m * self.embedding_size, activation=tf.nn.relu)
 
            
        initializer_GloUni = tf.initializers.GlorotUniform()
        if params['use_entity_embeddings']:
            entity_initializer = tf.initializers.GlorotUniform()
        else:
            entity_initializer = tf.constant_initializer(value=0.0)
            
            
        relation_shape=[self.action_vocab_size, 2 * self.embedding_size]
        
        self.relation_lookup_table = tf.Variable(initializer_GloUni(shape=relation_shape),
                                                         shape=relation_shape,
                                                         trainable=self.train_relations)
        if params['pretrained_embeddings_action'] != '':
            action_embedding = np.loadtxt(open(params['pretrained_embeddings_entity'] ))
            self.relation_lookup_table.assign(action_embedding)


        entity_shape= [self.entity_vocab_size, 2 * self.entity_embedding_size]
        
        self.entity_lookup_table = tf.Variable(entity_initializer(shape=entity_shape),
                                                   shape=entity_shape,
                                                   trainable=self.train_entities)
        if params['pretrained_embeddings_entity'] != '':
            entity_embedding = np.loadtxt(open(params['pretrained_embeddings_entity'] ))
            self.entity_lookup_table.assign(entity_embedding)
            
            
        # rnn_cells = [tf.keras.layers.LSTMCell(self.m * self.hidden_size) for _ in range(self.LSTM_Layers)]
        # self.policy_step1 = tf.keras.layers.StackedRNNCells(rnn_cells)
        
        cells = []
        for _ in range(self.LSTM_Layers):
            cells.append(tf.compat.v1.nn.rnn_cell.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
        #LSTM to generate the history
        self.policy_step = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        
        self.state_init = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        self.relation_init = self.dummy_start_label
    
    def get_query_embedding(self,query_relation):
        query_embedding = tf.nn.embedding_lookup(params=self.relation_lookup_table, ids=query_relation)  # [B, 2D]
        return query_embedding
    
    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    #feed forward network
    def policy_MLP(self, state):       
        hidden = self.dense1(state) 
        output = self.dense2(hidden) 
        return output

    #gets and concatenates the embeddings for a given relation and entity or set of them
    def action_encoder(self, next_relations, next_entities):
        with tf.compat.v1.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(params=self.relation_lookup_table, ids=next_relations)
            entity_embedding = tf.nn.embedding_lookup(params=self.entity_lookup_table, ids=next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    #all the relations for all the queries in the batch, all the next states for all the queries in the batch, all the previous histories for the queries in the batch, all the relatinos tht attached the 
    #previous states to the current states, all of the current entities for all the queries in the batch
    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
              range_arr, first_step_of_test):

        #concatenate last action/relation and current state ([at-1;ot] in the paper)
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        # 1. one step of rnn
        #get the new history. I am led to believe that output and new_state are the same, and are kept separate for data preservation reasons
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # output: [B, 4D]

        # Get state vector
        #get the embeddings of the current node
        prev_entity = tf.nn.embedding_lookup(params=self.entity_lookup_table, ids=current_entities)
        if self.use_entity_embeddings:
            #[ht;ot]
            state = tf.concat([output, prev_entity], axis=-1)
        else:
            state = output
        #generates matrix At, the stack of embeddings for all possible next actions
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        #[ht;ot;rq] which gets fed to the first layer of the FF network
        state_query_concat = tf.concat([state, query_embedding], axis=-1)

        # MLP for policy#

        #gets the output from the feed forward policy neural network
        output = self.policy_MLP(state_query_concat)
        #make the output have the same dimensions as the action matrix
        output_expanded = tf.expand_dims(output, axis=1)  # [B, 1, 2D]
        #multiply them together. This is the first step of creating the action probability distribution
        prelim_scores = tf.reduce_sum(input_tensor=tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Masking PAD actions
        #makes all actions marked as PAD impossible to choose. This prevents the agent from encountering the failure mode where it goes to an action that is correct but not desired.
        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf.compat.v1.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]

        # sample action
        #choose an action randomly from the distribution, getting the index and action label
        action = tf.cast(tf.random.categorical(logits=scores, num_samples=1), dtype=tf.int32)  # [B, 1]

        # loss
        #calculate the loss based on the uncertainty of the estimate (e.g. will be higher if multiple actions with similar probabilities)
        #label action is the same as action_idx
        label_action =  tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]

        # 6. Map back to true id
        #action idx is the index in the array of possible actions generated by the graph that the agent chose
        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(a=tf.stack([range_arr, action_idx])))

        #return loss, history, action probability distribution (logits), chosen action, and chosen relation
        return loss, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation, prelim_scores


