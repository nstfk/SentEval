# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
TREC question-type classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from senteval.tools.validation import KFoldClassifier


class PUBMED20KEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : PUBMED 20K *****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(task_path, 'train.txt'))
        self.test = self.loadFile(os.path.join(task_path, 'test.txt'))
        self.valid = self.loadFile(os.path.join(task_path, 'dev.txt'))

    def do_prepare(self, params, prepare):
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        data = {'X': [], 'y': []}
        tgt2idx = {'BACKGROUND': 0, 'OBJECTIVE': 1, 'METHODS': 2,
                   'RESULTS': 3, 'CONCLUSIONS': 4}
        with io.open(fpath, 'r', encoding='latin-1') as f:
            for line in f:
                try:
                  label, text = line.split('\t')
                  data['X'].append(text)
                  data['y'].append(tgt2idx[label])
                except:
                  pass
        #print(len(trec_data['X']),len(trec_data['y']))
        return data

    def run(self, params, batcher):
        train_embeddings, test_embeddings = [], []

        # Sort to reduce padding
        sorted_corpus_train = sorted(zip(self.train['X'], self.train['y']),
                                     key=lambda z: (len(z[0]), z[1]))
        train_samples = [x for (x, y) in sorted_corpus_train]
        train_labels = [y for (x, y) in sorted_corpus_train]
        
        sorted_corpus_valid = sorted(zip(self.valid['X'], self.valid['y']),
                                     key=lambda z: (len(z[0]), z[1]))
        valid_samples = [x for (x, y) in sorted_corpus_valid]
        valid_labels = [y for (x, y) in sorted_corpus_valid]

        sorted_corpus_test = sorted(zip(self.test['X'], self.test['y']),
                                    key=lambda z: (len(z[0]), z[1]))
        test_samples = [x for (x, y) in sorted_corpus_test]
        test_labels = [y for (x, y) in sorted_corpus_test]

        # Get train embeddings
        for ii in range(0, len(train_labels), params.batch_size):
            batch = train_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            train_embeddings.append(embeddings)
        train_embeddings = np.vstack(train_embeddings)
        logging.info('Computed train embeddings')
        
        # Get validation embeddings
        for ii in range(0, len(train_labels), params.batch_size):
            batch = valid_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            valid_embeddings.append(embeddings)
        train_embeddings = np.vstack(valid_embeddings)
        logging.info('Computed Validation embeddings')

        # Get test embeddings
        for ii in range(0, len(test_labels), params.batch_size):
            batch = test_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            test_embeddings.append(embeddings)
        test_embeddings = np.vstack(test_embeddings)
        logging.info('Computed test embeddings')
        
        config = {'nclasses': 5, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(X={'train': train_embeddings,
                                 'valid': valid_embeddings,
                                 'test': test_embeddings},
                              y={'train': train_labels,
                                 'valid': valid_labels,
                                 'test': test_labels},
                              config=config_classifier)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for SNLI\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
