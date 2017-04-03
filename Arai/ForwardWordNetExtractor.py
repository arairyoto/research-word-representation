# Python Version of WordNetExtractor.java
import os
import sys
#WordNet
import nltk
from nltk.corpus import WordNetCorpusReader
from nltk.corpus import wordnet as wn

import codecs

import util

class ForwardWordNetExtractor:
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
        if self.file_name.split(".")[-1] is "bin":
            self.Shared.loadGoogleModel(self.file_name);
        elif self.file_name.split(".")[-1] is "txt":
            self.Shared.loadTxtModel(self.file_name)
        else:
            self.Shared.loadModel(self.file_name)

        ver = wn.get_version()
        print("RESOURCE: WN " + str(ver) + "\n")
        print("LANGUAGE: "+self.lang+"\n")
        print("VECTORS: " + self.folder + "\n")
        print("TARGET: " + self.folder + "\n")

        self.extractWordsAndSynsets(self.folder + "words.txt",self.folder + "synsets.txt",self.folder + "lexemes.txt")
        self.extractSynsetRelations(self.folder + "hypernym.txt", '@')
        self.extractSynsetRelations(self.folder + "similar.txt",  '&')
        self.extractSynsetRelations(self.folder + "verbGroup.txt",  '$')
        # self.extractSynsetRelations(self.folder + "antonym.txt",  '!')

        print("DONE")

    def extractWordsAndSynsets(self, filenameWords, filenameSynsets,  filenameLexemes):
        #file
        fWords = codecs.open(filenameWords, 'w', 'utf-8')
        fSynsets = codecs.open(filenameSynsets, 'w', 'utf-8')
        fLexemes = codecs.open(filenameLexemes, 'w', 'utf-8')

        wordCounter = 0
        wordCounterAll = 0
        synsetCounter = 0
        synsetCounterAll = 0
        lexemCounter = 0
        lexemCounterAll = 0

        ovv = []

        for pos in self.pos_list:
            for synset in wn.all_synsets(pos=pos):
                synsetCounterAll += 1
                synsetId = synset.name()
                self.SynsetIndex[synsetId] = synsetCounterAll

                fSynsets.write(synsetId+" ")

                wordInSynset = 0

                for lemma in synset.lemmas():
                    lexemCounterAll += 1
                    wordId = lemma.name()

                    if self.Shared.in_vocab(wordId):
                        wordInSynset += 1
                        if wordId not in self.WordIndex:
                            fWords.write(wordId + " " + self.Shared.getVectorAsString(self.Shared.model[wordId]) + "\n")
                            wordCounter += 1
                            self.WordIndex[wordId] = wordCounter

                        lexemCounter += 1
                        #lemma name
                        sensekey = lemma.key()

                        fSynsets.write(sensekey + ",")
                        fLexemes.write(str(self.WordIndex[wordId]) + " " + str(synsetCounterAll) + "\n")
                    else:
                        ovv.append(wordId)

                fSynsets.write("\n")
                if wordInSynset is not 0:
                    synsetCounter += 1
                else:
                    self.SynsetIndex[synsetId] = -1
        fWords.close()
        fSynsets.close()
        fLexemes.close()

        print("   Words: %d / %d\n" % (wordCounter, wordCounter+len(ovv)))
        print("  Synset: %d / %d\n" % (synsetCounter, synsetCounterAll))
        print("  Lexems: %d / %d\n" % (lexemCounter, lexemCounterAll))

    def extractSynsetRelations(self, filename, relation_symbol):
        affectedPOS = {}

        f = codecs.open(filename, 'w', 'utf-8')

        for pos in self.pos_list:
            for synset in wn.all_synsets(pos=pos):
                synsetId = synset.name()
                targetSynsets = synset._related(relation_symbol)
                for targetSynset in targetSynsets:
                    targetSynsetId = targetSynset.name()
                    key = targetSynset.pos()

                    if key in affectedPOS:
                        affectedPOS[key] += 1
                    else:
                        affectedPOS[key] = 1

                    if self.SynsetIndex[synsetId] >= 0 and self.SynsetIndex[targetSynsetId] >= 0:
                        f.write(str(self.SynsetIndex[synsetId]))
                        f.write(" ")
                        f.write(str(self.SynsetIndex[targetSynsetId]))
                        f.write("\n")
        f.close()
        print("Extracted %s: done!\n" % self.pointer_map[relation_symbol])

        for k,v in affectedPOS.items():
            print("  %s: %d\n" % (k, v))

if __name__ == '__main__':
    #path to input word embeddings
    file_name = 'C:\\research_models\\wn_glosses.model'
    #path to output folder
    folder = 'Test\\'
    #language
    lang = 'eng'
    fwne = ForwardWordNetExtractor(file_name, folder, lang)
    fwne.main()
