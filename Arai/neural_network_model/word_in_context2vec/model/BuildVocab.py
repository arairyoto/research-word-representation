import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary

#Original
import NegativeSampling as ns

class Vocab(object):

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))

class BuildVocab:

    def build_vocab(self, model, articles, progress_per=10000):
        self.scan_vocab(model, articles, progress_per=progress_per)  # initial survey
        self.scale_vocab(model)  # trim by min_count & precalculate downsampling
        self.finalize_vocab(model)  # build tables & arrays

    #article = ([categories], sentence)
    def scan_vocab(self, articles, progress_per=10000):
        """Do an initial scan of all words appearing in articles."""
        logger.info("collecting all categories, words and their counts")
        article_no = -1
        total_words = 0
        total_categories = 0
        word_vocab = defaultdict(int)
        category_vocab = defaultdict(int)
        checked_string_types = 0

        for article_no, article in enumerate(articles):
            categories = article[0] #list
            sentence = str(article[1]).split(" ") #string

            if not checked_string_types:
                if isinstance(sentence, string_types):
                    logger.warn("Each 'sentences' item should be a list of words (usually unicode strings)."
                                "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
            if article_no % progress_per == 0:
                logger.info("PROGRESS: at article #%i, processed %i categories, keeping %i category types, %i words, keeping %i word types",
                            article_no, sum(itervalues(category_vocab)) + total_categories, len(category_vocab), sum(itervalues(word_vocab)) + total_words, len(word_vocab))

            for category in categories:
                category_vocab[category] += 1
            for word in sentence:
                word_vocab[word] += 1

        total_categories += sum(itervalues(category_vocab))
        total_words += sum(itervalues(word_vocab))

        logger.info("collected %i word types from a corpus of %i raw words and %i articles",
                    len(word_vocab), total_words, article_no + 1)
        model.corpus_count = article_no + 1
        model.raw_category_vocab = category_vocab
        model.raw_word_vocab = word_vocab

    def scale_vocab(self, model, min_count=None):

        min_count = min_count or model.min_count
        drop_ctotal = drop_cunique = 0
        drop_wtotal = drop_wunique = 0

        logger.info("Loading a fresh vocabulary")
        retain_ctotal, retain_categories = 0, []
        retain_wtotal, retain_words = 0, []
        # Discard words less-frequent than min_count
        model.cv.index2word = []
        model.wv.index2word = []
        # make stored settings match these applied settings
        model.min_count = min_count

        model.cv.vocab = {}
        model.wv.vocab = {}

        #Category Vocab
        for category, v in iteritems(model.raw_category_vocab):
            if keep_vocab_item(category, v, min_count):
                retain_categories.append(category)
                retain_ctotal += v

                model.cv.vocab[category] = Vocab(count=v, index=len(model.cv.index2word))
                model.cv.index2word.append(category)
            else:
                drop_cunique += 1
                drop_ctotal += v

        #Word Vocab
        for word, v in iteritems(model.raw_word_vocab):
            if keep_vocab_item(word, v, min_count):
                retain_words.append(word)
                retain_wtotal += v

                model.wv.vocab[word] = Vocab(count=v, index=len(model.wv.index2word))
                model.wv.index2word.append(word)
            else:
                drop_wunique += 1
                drop_wtotal += v

        original_cunique_total = len(retain_categories) + drop_cunique
        original_wunique_total = len(retain_words) + drop_wunique

        retain_cunique_pct = len(retain_categories) * 100 / max(original_cunique_total, 1)
        retain_wunique_pct = len(retain_words) * 100 / max(original_wunique_total, 1)

        logger.info("min_count=%d retains %i unique categories (%i%% of original %i, drops %i)",
                    min_count, len(retain_categories), retain_cunique_pct, original_cunique_total, drop_cunique)
        logger.info("min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
                    min_count, len(retain_words), retain_wunique_pct, original_wunique_total, drop_wunique)

        original_ctotal = retain_ctotal + drop_ctotal
        original_wtotal = retain_wtotal + drop_wtotal
        retain_cpct = retain_ctotal * 100 / max(original_ctotal, 1)
        retain_wpct = retain_wtotal * 100 / max(original_wtotal, 1)
        logger.info("min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                    min_count, retain_ctotal, retain_cpct, original_ctotal, drop_ctotal)
        logger.info("min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                    min_count, retain_wtotal, retain_wpct, original_wtotal, drop_wtotal)

        threshold_count = retain_wtotal

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = model.raw_word_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            model.wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        logger.info("deleting the raw counts category dictionary of %i items", len(self.raw_category_vocab))
        self.raw_category_vocab = defaultdict(int)
        logger.info("deleting the raw counts word dictionary of %i items", len(self.raw_word_vocab))
        self.raw_word_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_wtotal, 1), retain_wtotal)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_wunique, 'retain_total': retain_wtotal,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(model, vocab_size=len(retain_words))

        return report_values

    def finalize_vocab(self, model):
        #creating negative sampling table
        ns.make_cum_table(model)
        if model.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(model.wv.vocab)
            model.wv.index2word.append(word)
            model.wv.vocab[word] = v
        # set initial input/projection and hidden weights
        model.reset_weights()

    def estimate_memory(self, model, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size."""
        vocab_size = vocab_size or len(model.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * 500
        report['syn0'] = vocab_size * model.word_vector_size * dtype(REAL).itemsize

        if model.negative:
            report['syn1neg'] = vocab_size * model.word_layer1_size * dtype(REAL).itemsize

        report['total'] = sum(report.values())
        logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
                    vocab_size, model.word_vector_size, report['total'])
        return report
