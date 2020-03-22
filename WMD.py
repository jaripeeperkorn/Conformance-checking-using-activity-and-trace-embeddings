# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:45:26 2020

@author: u0132580
"""

import gensim
import pm4py
from gensim.corpora.dictionary import Dictionary
from pyemd import emd
import numpy as np
import copy

class WmDistance(object):
    def __init__(self, wv, docset1, docset2):
        self.wv = wv
        self.docset1 = docset1
        self.docset2 = docset2
        self.dists = np.full((len(self.docset1), len(self.docset2)), np.nan)
        self.dictionary = Dictionary(documents=self.docset1 + self.docset2)
        self.vocab_len = len(self.dictionary)
        self._cache_nbow()
        self._cache_dmatrix()
    def _cache_nbow(self):
        self.nbow1 = [self._nbow(doc) for doc in self.docset1]
        self.nbow2 = [self._nbow(doc) for doc in self.docset2]
    def _nbow(self, document):
        d = np.zeros(self.vocab_len, dtype=np.double)
        nbow = self.dictionary.doc2bow(document)
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)
        return d
    def _cache_dmatrix(self):
        self.distance_matrix = np.zeros((self.vocab_len, self.vocab_len), dtype=np.double)
        for i, t1 in self.dictionary.items():
            for j, t2 in self.dictionary.items():
                if self.distance_matrix[i, j] != 0.0: continue
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = \
                    np.sqrt(np.sum((self.wv[t1] - self.wv[t2])**2))
    def __getitem__(self, ij):
        if np.isnan(self.dists[ij[0], ij[1]]):
            self.dists[ij[0], ij[1]] = emd(self.nbow1[ij[0]], self.nbow2[ij[1]], self.distance_matrix)
        return self.dists[ij[0], ij[1]]
    
def get_dist(log1, log2, windowsize):   
    GT_log = copy.deepcopy(log1)
    pert_log = copy.deepcopy(log2)
    
    model = gensim.models.Word2Vec(GT_log + pert_log, size= 16, window=windowsize,  min_count=0, sg = 0)
    model.train(GT_log + pert_log, total_examples=len(GT_log + pert_log), epochs=100)
    
    #print("Model training done")
    
    WMD = WmDistance(model.wv, pert_log, GT_log)
    
    def distmatrix(GTlog, pertlog):
        distances = np.zeros((len(pertlog),len(GTlog)))
        for i in range(len(pertlog)):
            #if i % 50 == 0:
               # print ('Now calculating trace number %s'%i)
            for j in range(len(GTlog)):
                distances[i][j] = WMD[i,j]
        return distances
    
    disM = distmatrix(GT_log, pert_log)
    
    precision = np.average(disM.min(axis=1))
    fitness = np.average(disM.min(axis=0))
    
    return(precision, fitness)