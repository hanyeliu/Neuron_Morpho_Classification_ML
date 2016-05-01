# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

import os
import lvltrace
import sys
import shutil
import input_measure as im
import inputs
import time
import csv

# Added lib
import subprocess
import numpy as np
import multiprocessing
from multiprocessing import Pool

##############################################################
# Features extraction using L-Measure Command Line
##############################################################

# L-Measure Command Line with features
def lmeasure((neuron_features,neuron_path)):
    subprocess.call([inputs.l_measure_command,im.f0,im.f1,im.f2,im.f3,im.f4,im.f5,im.f6,im.f7,im.f8,im.f9,im.f10,im.f11,im.f12,im.f13,im.f14,im.f15,im.f16,im.f18,im.f19,im.f20,im.f21,im.f22,im.f23,im.f24,im.f25,im.f26,im.f27,im.f28,im.f29,im.f30,im.f31,im.f32,im.f33,im.f34,im.f35,im.f36,im.f37,im.f38,im.f39,im.f40,im.f41,im.f43,im.f44,neuron_features,neuron_path])

# Look for neurons in the input folder and extract features to the output folder
def features_extraction():
    lvltrace.lvltrace("LVLEntree dans features_extraction")
    data =[]
    for root, dirs, files in os.walk(inputs.morphology):
        for filename in files:
            if not filename.startswith('.'):
                if not os.path.isfile(filename) and filename.endswith('.swc'):
                    # neuron_path and neuron_features are the inputs of lmeasure
                    neuron_path=root+"/"+filename
 
                    neuron_features="-s"+os.path.splitext(os.path.splitext(neuron_path)[0])[0]+".csv"

                    # neuron_csv is the name of the extracted features output file from lmeasure
                    #neuron_csv=os.path.splitext(os.path.splitext(neuron_path)[0])[0]+".csv"
		    #print ("LVL neuron csv est "+ neuron_csv)

                    # Use of lmeasure tool to extract features and save it to input folders
		    data.append((neuron_features,neuron_path))

    cpuCount = multiprocessing.cpu_count()
    pool = Pool(processes=cpuCount)
    pool.map(lmeasure,data)
    lvltrace.lvltrace("LVLSortie dans features_extraction")













