# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:49:18 2020

@author: Kiryeong Nam
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
from collections import Counter 

start_scope()

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

w_input_hidden = pickle.load(open("w_data1.p", "rb"))
w_hidden_output = pickle.load(open("w_data2.p", "rb"))

w1 = numpy.array(w_input_hidden)
w2 = numpy.array(w_hidden_output)

w1 = w1.ravel() 
w2 = w2.ravel() 

w1 = np.exp(w1) - 1
w2 = np.exp(w2) - 1

duration = 1000*ms

z=0

for i in range(200):
    
    P = PoissonGroup(784, rates=x_test[i].flatten()*Hz)

    eqs = '''
          dv/dt=(I-v)/tau : 1
          I : 1
          tau : second
          '''
          
    G1 = NeuronGroup(100, eqs, threshold='v>10', reset='v=0', refractory = 5*ms, method = 'exact')
    G1.I = 0
    G1.tau = 20*ms

    S1 = Synapses(P, G1, 'w : 1', on_pre='v += w')
    S1.connect() 
    S1.w = w1
     
    G2 = NeuronGroup(10, eqs, threshold='v>3', reset='v=0', refractory = 5*ms, method = 'exact')
    G2.I = 0
    G2.tau = 20*ms
    
    M1 = SpikeMonitor(G1)
    
    S2 = Synapses(G1, G2, 'w : 1', on_pre='v += w')
    S2.connect()
    S2.w = w2
    
    M2 = SpikeMonitor(G2,name='output_spikes')
    run(duration)
      
    plt.plot(M2.t/ms, M2.i, '.k')
    xlim(0, 1000)   
    ylim(0, 10) 
    xlabel('Time (ms)')
    ylabel('Neuron index')
    
    count = Counter(M2.i)    
    if max(count, key = count.get) == y_test[i]:
        z = z + 1
    
print('accuracy : %.2f%%' %(z/200 * 100))

