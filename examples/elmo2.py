from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import torch
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.info("ELMO MODEL [ALLENNLP] (params: Path to Data , options_file[optional], weight file[optional] , Num of Hidden Layers[optional] ) ")
logging.info("\n\n\nPATH_TO_DATA: " + str(sys.argv[1])+ "\n\n")

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = sys.argv[1]  # '../data'


# import SentEval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False 

# Load ELMO model
#options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json'
#weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'


if (len(sys.argv)==4):
	elmo_encoder = ElmoEmbedder(sys.argv[2],sys.argv[3],cuda_device=0)
else:
	elmo_encoder = ElmoEmbedder(cuda_device=0)
params_senteval['elmo'] = elmo_encoder

if (len(sys.argv)==5):
    nhid = int(sys.argv[4])
else:
    nhid=0
params_senteval['classifier'] ={'nhid': nhid, 'optim': 'adam','batch_size': 64, 'tenacity': 5,'epoch_size': 4}




def s_embedding(word_embeds, rule='MEAN'):
    '''
    defines the type of sentence embedding
    @param word_embeds: word embeddings - np array of arrays
    @param rule: type of sentence embedding
    @return sentence_embedding
    '''
    if rule == 'MEAN':
        return np.mean(word_embeds, axis=0)

    if rule == 'SUM':
        return np.sum(word_embeds, axis=0)

    return 0

def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    return


def batcher(params, batch):
    """
    
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    #print(batch)

       
    #for elmo_embedding in params.elmo.embed_sentences(batch):  
    for elmo_embedding in params_senteval['elmo'].embed_sentences(batch):  
        # Average the 3 layers returned from ELMo #1024
        avg_elmo_embedding = np.average(elmo_embedding, axis=0)
        
        #concatenate the 3 layers returned from ELMo #3072
        comb_elmo_embedding = np.concatenate(elmo_embedding, axis=1)
        
        mowe_elmo=np.mean(comb_elmo_embedding, axis=0)   
        embeddings.append(mowe_elmo)
        
    embeddings = np.vstack(embeddings)
    return embeddings
  





if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    # here you define the NLP taks that your embedding model is going to be evaluated
    # in (https://arxiv.org/abs/1802.05883) we use the following :
    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['BIOSSES','ClinicalSTS']#,'MEDNLI']
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)
