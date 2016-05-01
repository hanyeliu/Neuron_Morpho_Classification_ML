# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

import tools
import lvltrace
import os
import csv
import math
import inputs

import numpy as np
from numpy import sort
from scipy.stats import scoreatpercentile
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.preprocessing import Imputer
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import neighbors, datasets, linear_model, svm
import statsmodels.api as sm
import pandas as pd
import operator
import time

def data_preprocessing_descriptive(Extracted_Features,Coma_Features,Corrected_Features):
    lvltrace.lvltrace("LVLEntree dans data_preprocessing_descriptive dans preproc_descriptive")
    tools.separate_coma(Extracted_Features,Coma_Features)
    for root, dirs, files in os.walk(Coma_Features):
        for i in files:
            if not i.startswith('.'):
                input_i=Coma_Features+i
                output_i=Corrected_Features+i
                lines=tools.file_lines(input_i)
                ncol=tools.file_col(input_i)
                if lines >= 2:
                    file = open(output_i, "w")
                    writer=csv.writer(file, lineterminator='\t')
                    
                    data = np.genfromtxt(input_i,delimiter=',')
                    X = data[1:, 2:]
                    neuron_type = np.genfromtxt(input_i,delimiter=',',dtype=None)
                    y = neuron_type[:, 0] # (class)

                    neuron_name = np.genfromtxt(input_i,delimiter=',',dtype=None)
                    z = neuron_name[:, 1] # Neuron names
                    
                    features = np.genfromtxt(input_i,delimiter=',',dtype=None)
                    w = features[0, :] # features names
                    
                    #Replace missing values 'nan' by column mean
                    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
                    imp.fit(X)
                    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
                    # Output replacement "Nan" values
                    Y=imp.transform(X)
                    #print i
                    #print Y.shape, y.shape,z.shape
                    #print Y.shape[1]
                    
                    ####################
                    for line in xrange(Y.shape[0]+1):
                        for colonne in xrange(Y.shape[1]+2):
                            if line == 0:
                                if colonne == 0:
                                    file.write("%s\t"%y[line])
                                else:
                                    if colonne == 1:
                                        file.write("%s\t"%z[line])
                                    else:
                                        file.write("%s\t"%w[colonne])
                            else:
                                if colonne == 0:
                                    file.write("%s\t"%y[line])
                                else:
                                    if colonne == 1:
                                        file.write("%s\t"%z[line])
                                    else:
                                        file.write("%f\t"%Y[line-1,colonne-2])
                        file.write("\n")
                    #########################
                else:
                    print "Only one morphology !!!"
                file.close()
    lvltrace.lvltrace("LVLSortie dans data_preprocessing_descriptive dans preproc_descriptive")


                






