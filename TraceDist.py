# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:14:01 2019

@author: Jari Peeperkorn
"""

import gensim
import numpy as np
import pm4py
from scipy import spatial
from gensim.models.doc2vec import TaggedDocument
from pm4py.algo.filtering.log.variants import variants_filter

def get_dist(log1, log2, windowsize):
    
    
    tags_GT_log = []
    tags_pert_log = []
    
    for j in range(len(log1)):
        ID = str()
        for i in range(len(log1[j])):
            ID = ID + log1[j][i].replace(" ", "")
        trace_id = [ID]
        tags_GT_log.append(trace_id)
        
    for j in range(len(log2)):
        ID = str()
        for i in range(len(log2[j])):
            ID = ID + log2[j][i].replace(" ", "")
        trace_id = [ID]
        tags_pert_log.append(trace_id)
        
    bothlog = log1 + log2
    taggedlog = []
    
    
    for j in range(len(bothlog)):
        eventlist = []
        ID = str()
        for i in range(len(bothlog[j])):
            ID = ID + bothlog[j][i].replace(" ", "")
        trace_id = [ID]
        td = TaggedDocument(bothlog[j], trace_id)
        taggedlog.append(td)
    

    
    #use a combination of both logs to train, but each variant only once
    model = gensim.models.Doc2Vec(taggedlog, alpha=0.025, vector_size= 16, window=windowsize,  min_count=1, dm = 0)
    model.train(taggedlog, total_examples=len(taggedlog), epochs=300)
    
    #print("Model training done")
    
    def cosdis(trace1, trace2):
        rep1 = model.docvecs[trace1[0]]
        rep2 = model.docvecs[trace2[0]]
        return spatial.distance.cosine(rep1, rep2)
    
    def distmatrix(GTlog, pertlog):
        distances = np.full((len(pertlog),len(GTlog)), 1.0) #each trace of the perturbed log is a row and each column is a trace from GT
        for i in range(len(pertlog)):
            #if i % 50 == 0:
               # print ('Now calculating trace number %s'%i)
            for j in range(len(GTlog)):
                distances[i][j] = cosdis(pertlog[i],GTlog[j])
        return distances
    
    disM = distmatrix(tags_GT_log, tags_pert_log)
    #print(disM)
    
    precision = np.average(np.amin(disM, axis=1)) #average of the minima of each row = compare pert to GT
    #fitness = 0
    fitness = np.average(np.amin(disM, axis=0)) #aevrage of the minima of each column = compare GT to pert
    
    #print(np.amin(disM, axis=0))
    
    return(precision, fitness)