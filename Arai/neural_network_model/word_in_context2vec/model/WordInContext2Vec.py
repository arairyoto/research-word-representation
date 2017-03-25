import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools

from gensim.utils import keep_vocab_item, call_on_class_only
from gensim.utils import keep_vocab_item
from gensim.models.keyedvectors import KeyedVectors

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

from scipy.special import expit
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType
from scipy import stats
from queue import Queue, Empty

logger = logging.getLogger(__name__)

from SkipGram import SkipGram
from BuildVocab import BuildVocab
from NegativeSampling import NegativeSampling

sg = SkipGram()
bv = BuildVocab()
ns = NegativeSampling()

class WIC2Vec:

    def __init__(
            self, articles=None, category_size = 10, word_size=20, alpha=0.025, window=3, min_count=0,
            max_vocab_size=None, seed=1, workers=3, min_alpha=0.0001, negative=5, hashfxn=hash, iter=5, null_word=0):

        self.initialize_word_vectors()
        self.cum_table = None  # for negative sampling
        self.category_vector_size = int(category_size)
        self.category_layer1_size = int(category_size)
        self.word_vector_size = int(word_size)
        self.word_layer1_size = int(word_size)
        if word_size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        #学習率
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        #negative samplingの個数
        self.negative = negative
        self.iter = int(iter)
        self.hashfxn = hashfxn
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.model_trimmed_post_training = False
        self.neg_labels = []
        if self.negative > 0:
            # precompute negative labels optimization for pure-python training
            self.neg_labels = zeros(self.negative + 1)
            self.neg_labels[0] = 1
        if articles is not None:
            bv.build_vocab(self, articles)
            self.train(articles)

    def initialize_word_vectors(self):
        #word
        self.wv = KeyedVectors()
        #category
        self.cv = KeyedVectors()

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.wv.syn0 = empty((len(self.wv.vocab), self.word_vector_size, self.category_vector_size), dtype=REAL)
        self.cv.syn0 = empty((len(self.cv.vocab), self.category_vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.cv.vocab)):
            # construct deterministic seed from word AND seed argument
            self.cv.syn0[i] = self.seeded_vector(self.cv.index2word[i] + str(self.seed), self.category_vector_size)
        # print(self.cv.syn0)

        for i in xrange(len(self.wv.vocab)):
            # construct deterministic seed from word AND seed argument
            for j in xrange(self.word_vector_size):
                self.wv.syn0[i][j] = self.seeded_vector(self.wv.index2word[i] + str(j) + str(self.seed), self.category_vector_size)
        print(self.wv.syn0)

        if self.negative:
            self.syn1neg = zeros((len(self.wv.vocab), self.word_layer1_size, self.category_layer1_size), dtype=REAL)
        self.wv.syn0norm = None

        self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning

    def seeded_vector(self, seed_string, vector_size):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(vector_size) - 0.5) / vector_size

    def train(self, articles):
        bv.build_vocab(self, articles)
        for i in xrange(self.iter):
            sg.train_batch_sg(self, articles, self.alpha)


articles = [(["a","b","c"], "i am man woman man get go"), (["b","c","d"], "get go sam meaning men")]
w = WIC2Vec()
w.train(articles)

print(w.wv.syn0)
# w.train(articles)
