# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import sys
print(sys.argv[1:])
# get models.py from InferSent repo
from models import InferSent

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = sys.argv[1]#'../data'
PATH_TO_W2V =  sys.argv[2]#'fasttext/crawl-300d-2M.vec'# 'glove/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = sys.argv[3] #'infersent2.pkl'
V = int(sys.argv[4]) #2 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    logging.info("PATH_TO_DATA: " + str(PATH_TO_DATA) +"\nPATH_TO_W2V: "+ str(PATH_TO_W2V) + "\nMODEL_PATH: "+str(MODEL_PATH))

    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)

    params_senteval['infersent'] = model.cuda()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SNLI']
    results = se.eval(transfer_tasks)
    print(results)
