# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:15:52 2020

@author: u0132580
"""

import numpy as np
import copy

import WMD
import TraceDist
import ICT

from py4j.java_gateway import JavaGateway
from collections import deque

class OutputGetter(deque):
    def appendleft(self, line):
        print(line)
        super().appendleft(line)
        
def remove_plus(dummylog):
    newlog = copy.deepcopy(dummylog)
    newlog = [[item.replace('+', '') for item in lst] for lst in newlog]
    return newlog
        
def fix_duplicates(dummylog):
    newlog = []
    for trace in dummylog:
        newtrace = trace[::2]
        newlog.append(newtrace)
    return newlog

def check_duplicates(dummylog): #not a good test but just to be sure
    if dummylog[0][0] == dummylog[0][1] and dummylog[1][0] == dummylog[1][1]:
        return True
    else:
        return False
    

def perform_test(reallog, modelfilename, windowsize):
    
    gateway = JavaGateway.launch_gateway(classpath="./jars/*", 
                                         redirect_stdout=OutputGetter())
    petri_and_marking = gateway.jvm.org.processmining.plugins.kutoolbox.utils.ImportUtils.openPetrinet(
        gateway.jvm.java.io.File(modelfilename)
    )
    if not petri_and_marking[1].size():
        print("Creating initial marking")
        petri_and_marking[1] = gateway.jvm.org.processmining.plugins.kutoolbox.utils.PetrinetUtils. \
            getInitialMarking(petri_and_marking[0])
        
    petri_and_marking[1].size()
    settings = gateway.jvm.org.processmining.plugins.loggenerator.utils.GeneratorSettings()
    simulator = gateway.jvm.org.processmining.plugins.loggenerator.utils.GeneratorSettings.SimulationAlgorithm
    
    # See https://github.com/Macuyiko/processmining-prom/blob/master/loggenerator/
    #             org/processmining/plugins/loggenerator/utils/GeneratorSettings.java
    
    for t in petri_and_marking[0].getTransitions().iterator():
        isInvisible = t.getLabel() == "" or t.isInvisible()
        label = t.getLabel()
        isTextInvisible = label.startswith("inv_") or "$" in label
        mapped = label.replace("+complete", "").replace("\\ncomplete", "").replace("\\n", "") \
            if not isInvisible and not isTextInvisible else ""
        arr = gateway.new_array(gateway.jvm.int, 4)
        arr[0], arr[1], arr[2], arr[3] = 60, 0, 60, 0
        settings.getTransitionNames().put(t, mapped)
        settings.getTransitionWeights().put(t, 10)
        settings.getTransitionTimings().put(t, arr)
        #print(t.getId(), ':', label, '--->', mapped)
        
        settings.setNrTraces(1000)
        settings.setRandomMinInGroup(1)
        settings.setRandomMaxInGroup(1)
        settings.setMustReachEnd(True)
        settings.setMustConsumeAll(False)
        settings.setMaxTimesMarkingSeen(3)
        
        settings.setSimulationMethod(simulator.Random) # Random, Complete, or Distinct
        
        # settings.setRestartAfter(1000)
        # settings.setSkipChance(0.85)
        # settings.setAlsoConsiderPartial(False)
    xlog = gateway.jvm.org.processmining.plugins.loggenerator.PetriNetLogGenerator.generate(
        petri_and_marking[0], 
        petri_and_marking[1], 
        settings, 
        #None # or for output: gateway.jvm.org.processmining.plugins.kutoolbox.eventlisteners.PluginEventListenerCLI()
        gateway.jvm.org.processmining.plugins.kutoolbox.eventlisteners.PluginEventListenerCLI()
    )
    
    temp_log = []
    for trace in xlog:
        #print([gateway.jvm.org.deckfour.xes.extension.std.XConceptExtension.instance().extractName(e) for e in trace])
        temp_log.append([gateway.jvm.org.deckfour.xes.extension.std.XConceptExtension.instance().extractName(e) for e in trace])
    
    if check_duplicates(temp_log) == True:
        print('Fixing duplicates')
        temp_log = fix_duplicates(temp_log)
    modellog = remove_plus(temp_log)
    
    print("WMD:")
    WMD1, WMD2 = WMD.get_dist(reallog, modellog, windowsize)
    print("Fitness:", WMD2)
    print("Precision", WMD1)
    print("ICT:")
    ICT1, ICT2 = ICT.get_dist(reallog, modellog, windowsize)
    print("Fitness:", ICT2)
    print("Precision", ICT1)
    print("Trace2vec:")
    T1, T2 = TraceDist.get_dist(reallog, modellog, windowsize)
    print("Fitness:", T2)
    print("Precision", T1)
    