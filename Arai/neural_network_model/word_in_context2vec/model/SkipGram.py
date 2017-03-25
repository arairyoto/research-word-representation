import sys
import os

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

from scipy.special import expit

class SkipGram:
    def train_batch_sg(self, model, articles, alpha, work=None):
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
                            self.train_sg_pair(model, model.cv.index2word[category.index], model.wv.index2word[word.index], word2.index, alpha)
                result += len(word_vocabs)
        return result

    def train_sg_pair(self, model, category, word, context_index, alpha):
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

        # lock_factor = context_locks[context_index]

        # neu1e = zeros(l1.shape)

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
        model.cv.syn0[predict_category] += dot(back_context, category2context[context_index].T)

        #word learning
        for index, word_indice in enumerate(word_indices):
            model.syn1neg[word_indice] += outer(back_word[index], l0)
            model.cv.syn0[predict_category] += dot(back_word[index], category2word[word_indice].T)
