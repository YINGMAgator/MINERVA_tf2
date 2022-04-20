from collections import defaultdict
import logging
from urllib.parse import MAX_CACHE_SIZE
import numpy as np
# import csv

logger = logging.getLogger(__name__)


class RelationEntityGrapher:
    #UNDERSTOOD
    def __init__(self, triple_store, relation_vocab, entity_vocab, max_num_actions):

        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.triple_store = triple_store
        self.relation_vocab = relation_vocab
        self.entity_vocab = entity_vocab
        self.store = defaultdict(list)
        self.max_num_actions=max_num_actions
        #create graph: length=nodes, holds up to max_num_actions of edges where an edge is [e2,r]
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32'))
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.masked_array_store = None

        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        self.create_graph()
        print("KG constructed")

    #UNDERSTOOD
    def create_graph(self):
        # with open(self.triple_store) as triple_file_raw:
            # triple_file = csv.reader(triple_file_raw, delimiter='\t')

        #creates a dictionary of outgoing edges from a node by reading from the knowledge base
        for line in self.triple_store:
            e1 = self.entity_vocab[line[0]]
            r = self.relation_vocab[line[1]]
            e2 = self.entity_vocab[line[2]]
            self.store[e1].append((r, e2))


        for e1 in self.store:
            num_actions = 1
            #edge 0 at node e1: relation is NO_OP and destination node is e1
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[e1, 0, 0] = e1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:
                    print("The maximum number of connections to this node ("+str(self.max_num_actions)+") has been exceeded. "+str(len(self.store[e1])-200)+" connecting nodes are being ignored")
                    break
                #edge n at node e1: relation is relation r and destination node is e2
                self.array_store[e1,num_actions,0] = e2
                self.array_store[e1,num_actions,1] = r
                num_actions += 1
        del self.store
        self.store = None

    #UNDERSTOOD
    #receives a batch of states: (et, e1, rq, e2) for every query in the batch being pursued
    def return_next_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts):
        ret = self.array_store[current_entities, :, :].copy()
        #for query in batch
        for i in range(current_entities.shape[0]):
            #basically means we aren't allowed to go straight from the start node to the end in just one edge; we want longer logic chains I guess
            #if we are at the start node of query i
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                #generate a vector of length equal to number of edges from entity, where value is true if relation==query relation and e2=query answer for a given entity, false otherwise
                mask = np.logical_and(relations == query_relations[i] , entities == answers[i])
                #for every e2 and r, if that action is true in the mask, set the values of r and of e2 to the pad value for entity and relation respectively
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD
            if last_step:
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]

                #TURNED OFF BECAUSE WE DONT WANT TO MASK CORRECT ANSWERS ANYMORE
                #if an entity connected to the current node is a correct answer but not the correct answer, set it and the relation connecting it to the current node to the PAD value
                # correct_e2 = answers[i]
                # for j in range(entities.shape[0]):
                #     #print(i/rollouts,j,i,rollouts)
                #     if entities[j] in all_correct_answers[int(i/rollouts)] and entities[j] != correct_e2:
                #         entities[j] = self.ePAD
                #         relations[j] = self.rPAD

        return ret
