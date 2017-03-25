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

FAST_VERSION = -1
MAX_WORDS_IN_BATCH = 10000

class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))

def train_batch_sg(model, articles, alpha, work=None):
    result = 0
    #article = [(categories, sentence), ...]
    for article in articles:
        categories = article[0]
        sentence = str(article[1]).split(" ") #str to list

        category_vocabs = [model.cv.vocab[c] for c in categories]
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]

        for category in category_vocabs:
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        train_sg_pair(model, model.cv.index2word[category.index], model.wv.index2word[word.index], word2.index, alpha)
            result += len(word_vocabs)
    return result

def train_sg_pair(model, category, word, context_index, alpha):
    category2context = model.wv.syn0 #category vector to context vector
    category2word = model.syn1neg #category vector to word vector
    category_vectors = model.cv.syn0 #category index to category vector

    context_locks = model.syn0_lockf #?


    if category not in model.cv.vocab:
        return
    predict_category = model.cv.vocab[category]  # target category

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word

    l0 = category_vectors[predict_category.index]

    l1 = dot(category2context[context_index], l0)  # input word (NN input/projection layer)

    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
    #word_indices = [predict_word.index, ns1.index, ns2.index, ...]
    word_indices = [predict_word.index]
    while len(word_indices) < model.negative + 1:
        w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
        if w != predict_word.index:
            word_indices.append(w)

    l2b = zeros((len(word_indices), l1.shape[0])) # word_size * (negative + 1)
    for index, word_indice in enumerate(word_indices):
        l2b[index] = dot(category2word[word_indice], l0)

    fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
    gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate

    back_context = dot(gb, l2b)
    back_word = outer(gb, l1)

    #contect learning
    model.wv.syn0[context_index] += outer(back_context, l0)
    model.cv.syn0[predict_category.index] += dot(back_context, category2context[context_index])

    #word learning
    for index, word_indice in enumerate(word_indices):
        model.syn1neg[word_indice] += outer(back_word[index], l0)
        model.cv.syn0[predict_category.index] += dot(back_word[index], category2word[word_indice])

class WIC2Vec(utils.SaveLoad):

    def __init__(
            self, articles=None, category_size = 10, word_size=20, alpha=0.025, window=3, min_count=0,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=1, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):

        self.initialize_word_vectors()
        self.sg = int(sg)
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
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        #negative samplingの個数
        self.negative = negative
        self.hs = 0
        self.iter = int(iter)
        self.hashfxn = hashfxn
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        self.neg_labels = []
        if self.negative > 0:
            # precompute negative labels optimization for pure-python training
            self.neg_labels = zeros(self.negative + 1)
            self.neg_labels[0] = 1
        if articles is not None:
            self.build_vocab(articles)
            self.train(articles)

    def initialize_word_vectors(self):
        #word
        self.wv = KeyedVectors()
        #category
        self.cv = KeyedVectors()

    #same as word2vec
    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.
        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.
        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.wv.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += self.wv.vocab[self.wv.index2word[word_index]].count**power
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += self.wv.vocab[self.wv.index2word[word_index]].count**power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def build_vocab(self, articles, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        """
        self.scan_vocab(articles, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)  # trim by min_count & precalculate downsampling
        self.finalize_vocab(update=update)  # build tables & arrays

    #article = ([categories], sentence)
    def scan_vocab(self, articles, progress_per=10000, trim_rule=None):
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

            # if self.max_vocab_size and len(vocab) > self.max_vocab_size:
            #     total_words += utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
            #     min_reduce += 1

        total_categories += sum(itervalues(category_vocab))
        total_words += sum(itervalues(word_vocab))

        logger.info("collected %i word types from a corpus of %i raw words and %i articles",
                    len(word_vocab), total_words, article_no + 1)
        self.corpus_count = article_no + 1
        self.raw_category_vocab = category_vocab
        self.raw_word_vocab = word_vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None, update=False):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).
        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.
        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.
        """
        min_count = min_count or self.min_count
        drop_ctotal = drop_cunique = 0
        drop_wtotal = drop_wunique = 0

        logger.info("Loading a fresh vocabulary")
        retain_ctotal, retain_categories = 0, []
        retain_wtotal, retain_words = 0, []
        # Discard words less-frequent than min_count
        if not dry_run:
            self.cv.index2word = []
            self.wv.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample

            self.cv.vocab = {}
            self.wv.vocab = {}

        #Category Vocab
        for category, v in iteritems(self.raw_category_vocab):
            if keep_vocab_item(category, v, min_count, trim_rule=trim_rule):
                retain_categories.append(category)
                retain_ctotal += v
                if not dry_run:
                    self.cv.vocab[category] = Vocab(count=v, index=len(self.cv.index2word))
                    self.cv.index2word.append(category)
            else:
                drop_cunique += 1
                drop_ctotal += v

        #Word Vocab
        for word, v in iteritems(self.raw_word_vocab):
            if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                retain_words.append(word)
                retain_wtotal += v
                if not dry_run:
                    self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                    self.wv.index2word.append(word)
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


        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_wtotal
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_wtotal
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_word_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run:
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
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values

    def finalize_vocab(self, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.wv.index2word:
            self.scale_vocab()
        # if self.sorted_vocab and not update:
        #     self.sort_vocab()
        # if self.hs:
        #     # add info about each word's Huffman encoding
        #     self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.wv.vocab)
            self.wv.index2word.append(word)
            self.wv.vocab[word] = v
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights()
        else:
            self.update_weights()

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size."""
        vocab_size = vocab_size or len(self.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * 500
        report['syn0'] = vocab_size * self.word_vector_size * dtype(REAL).itemsize

        if self.negative:
            report['syn1neg'] = vocab_size * self.word_layer1_size * dtype(REAL).itemsize

        report['total'] = sum(report.values())
        logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
                    vocab_size, self.word_vector_size, report['total'])
        return report

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
        for i in xrange(self.iter):
            train_batch_sg(self, articles, self.alpha)


    #****************************************************************************

    def category_vector(self, category):
        return self.cv.syn0[self.cv.vocab[category].index]

    def word_vector(self, category, word):
        category_vector = self.cv.syn0[self.cv.vocab[category].index]
        converter = self.syn1neg[self.wv.vocab[word].index]
        return dot(converter, category_vector)

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors, recalculable table
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'table', 'cum_table'])

        super(WIC2Vec, self).save(*args, **kwargs)

    save.__doc__ = utils.SaveLoad.save.__doc__


#*************************************TEST***************************************

# articles = [(["a","b","c"], "i am man woman man get go"), (["b","c","d"], "get go sam meaning men")]

#Input and Save
# w = WIC2Vec(articles = articles)
# w.save("test2.model")

#Load and Try
# w2 = WIC2Vec.load("test2.model")
# print(w2.word_vector('a', 'woman'))
