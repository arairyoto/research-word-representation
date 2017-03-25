import logging
import sys
import os

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

class NegativeSampling:

    #NegativeSampling時に用いる確率分布
    def make_cum_table(self, model, power=0.75, domain=2**31 - 1):
        vocab_size = len(model.wv.index2word)
        model.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += model.wv.vocab[model.wv.index2word[word_index]].count**power
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += model.wv.vocab[model.wv.index2word[word_index]].count**power
            model.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(model.cum_table) > 0:
            assert model.cum_table[-1] == domain
