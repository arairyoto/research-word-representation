# Python Version of WordNetExtractor.java
import os
import sys
#WordNet
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import codecs

import util

def normalize(lemma):
    return lemma.replace("_", " ").replace("")

class BackwardWordNetExtractor:
    def __init__(self, file_name, folder, lang):
        self.file_name = file_name
        self.folder = folder
        if lang in wn.langs():
            self.lang = lang
        else:
            print("language: '%s' is not supported, try another language" % lang)
        #initialize
        self.WordIndex = {}
        self.SynsetIndex = {}
        self.pos_list = ['a', 's', 'r', 'n', 'v']
        self.pointer_map = {"@":"hypernym", "&":"similar", "$":"verbGroup", "!":"antonym"}
        self.Shared = util.Shared()

    def main(self):
        # if self.file_name.split(".")[-1] is "bin":
        #     self.Shared.loadGoogleModel(self.file_name)
        # elif self.file_name.split(".")[-1] is "txt":
        #     self.Shared.loadTxtModel(self.file_name)
        # else:
        #     self.Shared.loadModel(self.file_name)
        self.Shared.loadTxtModel(self.file_name)
        ver = wn.get_version()
        print("RESOURCE: WN " + str(ver) + "\n")
        print("LANGUAGE: "+self.lang+"\n")
        print("VECTORS: " + self.folder + "\n")
        print("TARGET: " + self.folder + "\n")
        self.extractWordsAndSynsets(self.folder + "words.txt",self.folder + "synsets.txt",self.folder + "lexemes.txt")
        self.extractWordRelations(self.folder + "hypernym.txt", '@')
        self.extractWordRelations(self.folder + "similar.txt",  '&')
        self.extractWordRelations(self.folder + "verbGroup.txt",  '$')
        # self.extractWordRelations(self.folder + "antonym.txt",  '!')

        print("DONE")


    def extractWordsAndSynsets(self, filenameWords, filenameSynsets,  filenameLexemes):
        #file
        fWords = codecs.open(filenameWords, 'w', 'utf-8')
        fSynsets = codecs.open(filenameSynsets, 'w',  'utf-8')
        fLexemes = codecs.open(filenameLexemes, 'w',  'utf-8')

        wordCounter = 0
        wordCounterAll = 0
        synsetCounter = 0
        synsetCounterAll = 0
        lexemCounter = 0
        lexemCounterAll = 0

        ovv = []

        for pos in self.pos_list:
            for word in wn.all_lemma_names(pos=pos, lang=self.lang):
                wordCounterAll += 1
                self.WordIndex[word] = wordCounterAll
                fWords.write(word+" ")
                synsetInWord = 0
                for synset in wn.synsets(word, lang=self.lang):
                    lexemCounterAll += 1
                    synsetId = synset.name()
                    if self.Shared.in_vocab(synsetId):
                        synsetInWord += 1
                        if synsetId not in self.SynsetIndex:
                            fSynsets.write(synsetId + " " + self.Shared.getVectorAsString(self.Shared.model[synsetId]) + "\n")
                            synsetCounter += 1
                            self.SynsetIndex[synsetId] = synsetCounter

                        lexemCounter += 1
                        #lemma name
                        sensekey = wn.lemma(synset.name()+'.'+word).key()

                        fWords.write(sensekey + ",")
                        fLexemes.write(str(self.SynsetIndex[synsetId]) + " " + str(wordCounterAll) + "\n")
                    else:
                        ovv.append(synsetId)


                fWords.write("\n")
                if synsetInWord is not 0:
                    wordCounter += 1
                else:
                    self.WordIndex[word] = -1
        fWords.close()
        fSynsets.close()
        fLexemes.close()
        print("   Words: %d / %d\n" % (wordCounter, wordCounterAll))
        print("  Synset: %d / %d\n" % (synsetCounter, synsetCounter + len(ovv)))
        print("  Lexems: %d / %d\n" % (lexemCounter, lexemCounterAll))

    def extractWordRelations(self, filename, relation_symbol):
        affectedPOS = {}
        f = codecs.open(filename, 'w', 'utf-8')
        for pos in self.pos_list:
            for synset in wn.all_synsets(pos=pos):
                targetSynsets = synset._related(relation_symbol)
                for targetSynset in targetSynsets:
                    for lemma in synset.lemmas(lang = self.lang):
                        word = lemma.name().lower()
                        for targetLemma in targetSynset.lemmas(lang = self.lang):
                            targetWord = targetLemma.name().lower()
                            key = targetLemma.synset().pos()

                            if key in affectedPOS:
                                affectedPOS[key] += 1
                            else:
                                affectedPOS[key] = 1

                            if word in self.WordIndex and targetWord in self.WordIndex:
                                if self.WordIndex[word] >= 0 and self.WordIndex[targetWord] >= 0:
                                    f.write(str(self.WordIndex[word]))
                                    f.write(" ")
                                    f.write(str(self.WordIndex[targetWord]))
                                    f.write("\n")
                            else:
                                print(word, targetWord)
        f.close()
        print("Extracted %s: done!\n" % self.pointer_map[relation_symbol])

        for k,v in affectedPOS.items():
            print("  %s: %d\n" % (k, v))

if __name__ == '__main__':
    #path to input word embeddings
    file_name = 'C:\\research_models\\AutoExtend\\synsets.txt'
    #path to output folder
    folder = 'Test_back\\'
    #language
    lang = 'eng'
    #List of Languages
    #     ['als', 'arb', 'cat', 'cmn', 'dan', 'eng', 'eus', 'fas',
    # 'fin', 'fra', 'fre', 'glg', 'heb', 'ind', 'ita', 'jpn', 'nno',
    # 'nob', 'pol', 'por', 'spa', 'tha', 'zsm']
    bwne = BackwardWordNetExtractor(file_name, folder, lang)
    bwne.main()
