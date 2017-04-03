import os
import sys

import numpy as np
from gensim.models import word2vec

import codecs

class Shared:
    def loadModel(self, file_name):
        self.model = word2vec.Word2Vec.load(file_name)
        self.is_w2v = True

    def loadGoogleModel(self, file_name):
        self.model = word2vec.Word2Vec.load_word2vec_format(file_name, binary=True)
        self.is_w2v = True

    def loadTxtModel(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        line = f.readline()
        lines = []

        while line:
            lines.append(line)
            line = f.readline()
        f.close()
        model = {}

        for idx, l in enumerate(lines[1:]):
            #最後の改行を除いてスペースでスプリット
            temp = l.replace("\n", "").split(" ")
            word = temp[0]
            # word = temp[0].split("-")[2:]
            # synset = wnm.wn._synset_from_pos_and_offset(word[1], int(word[0]))
            embedding = [float(x) for x in temp[1:]]
            model[word] = embedding
            # dic[synset.name()] = embedding
        self.model = model
        self.is_w2v = False

    def in_vocab(self, key):
        if self.is_w2v:
            if key in self.model.wv.vocab:
                return True
            else:
                return False
        else:
            if key in self.model:
                return True
            else:
                return False


    def getVectorAsString(self, vector):
        sb = ""
        for i in range(len(vector)):
            sb += str(vector[i])
            sb += " "
        return sb.strip()
