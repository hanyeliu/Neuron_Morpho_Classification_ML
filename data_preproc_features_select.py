# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

import tools
import lvltrace
import os
import numpy as np
from numpy import sort
import csv
from scipy.stats import scoreatpercentile
import math
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
#import statsmodels.api as sm
import pandas as pd
import operator
import time

#################################################################################################

#                                      Data Preprocessing                                       #

#################################################################################################

# Preprocess the extracted features -- standardization, missing values, encoding categorial features (class)
# Prepocessing with all features
def preprocessing_module(Extracted_Features,Coma_Features,Corrected_Features, Norm,ontology):
    # Replace tab separated csv into comma separated csv and replace categorial variables into iteration
    lvltrace.lvltrace("LVLEntree dans preprocessing_module data_preproc")
    onto = open(ontology, "w")
    writer=csv.writer(onto, lineterminator=',')
    class_number = 1
    onto.write("Iteration,Class,Class_number,Neuron_name\n")
    Iteration=1
    for root, dirs, files in os.walk(Extracted_Features):
        for i in files:
            if not i.startswith('.'):
                #print i
                input_i=Extracted_Features+i
                output_i=Coma_Features+i
                file = open(output_i, "w")
                writer=csv.writer(file, lineterminator=',')
                lines=tools.file_lines(input_i)+1
                ncol=tools.file_col(input_i)-1
                for line in xrange(lines):
                    for col in xrange(ncol):
                        if line == 0:
                            if col == 1: # Skipping neuron names
                                laurent=1
                            else:
                                file.write("%s,"%tools.read_csv_tab(input_i,col,line))
                        else:
                            if col == 0: # replace class names by an integer
                                file.write("%i,"%class_number)
                            
                            else:
                                if col == 1:
                                    #print "skip neuron name"
                                    onto.write("%i,%s,%i,%s\n"%(Iteration,i,class_number,tools.read_csv_tab(input_i,col,line)))
                                    Iteration=Iteration+1
                                else:
                                    file.write("%s,"%tools.read_csv_tab(input_i,col,line))
                    file.write("\n")
                file.close()
                class_number = class_number + 1
                if lines > 3 :
                    input_file=Coma_Features+i
                    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1),skiprows=1) # ncol-1 because we skip the class names
                    X = data[:, :ncol]
                    y = data[:, 0].astype(np.int) # Labels (class)
                    #Replace missing values 'nan' by column mean
                    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
                    imp.fit(X)
                    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
                    # Output replacement "Nan" values
                    Y=imp.transform(X)
                    #Data Standardization
                    if Norm == 'normalize':
                        Z=preprocessing.normalize(Y, axis=0, norm='l2') # Normalize
                    else:
                        if Norm == 'binarize':
                            binarizer=preprocessing.Binarizer().fit(Y) # Binarize for Bernoulli
                            Z = binarizer.transform(Y)
                        else:
                            if Norm == 'standardize':
                                min_max_scaler = preprocessing.MinMaxScaler() # Normalize the data to [0,1]
                                Z=min_max_scaler.fit_transform(Y)
                            else:
                                Z=preprocessing.scale(Y) #Scaling

                    #Create new files with corrected and standardized data
                    output_file=Corrected_Features+i
                    file = open(output_file, "w")
                    writer=csv.writer(file, lineterminator=',')
                    for line_1 in xrange(lines-1):
                        for col_1 in xrange(ncol-1):
                            if col_1==0:
                                file.write("%s,"%y[line_1])
                            else:
                                file.write("%f,"%Z[line_1,col_1])
                        file.write("\n")
                    file.close()
                else:
                    laurent=1
    onto.close()
    lvltrace.lvltrace("LVLSortie dans preprocessing_module data_preproc")

# Prepocessing with selected features
def preprocessing_module_features_selected(Extracted_Features,Coma_Features,Corrected_Features,Norm):
    # Replace tab separated csv into comma separated csv and replace categorial variables into iteration
    onto = open(ontology, "w")
    writer=csv.writer(onto, lineterminator=',')
    class_number = 1
    onto.write("Iteration,Class,Class_number,Neuron_name\n")
    Iteration=1
    for root, dirs, files in os.walk(Extracted_Features):
        for i in files:
            if not i.startswith('.'):
                input_i=Extracted_Features+i
                output_i=Coma_Features+i
                file = open(output_i, "w")
                writer=csv.writer(file, lineterminator=',')
                lines=tools.file_lines(input_i)+1
                ncol=tools.file_col(input_i)
                for line in xrange(lines):
                    for col in xrange(ncol):
                        if line == 0:
                            if col == 1: # Skipping neuron names
                                laurent=1
                            else:
                                file.write("%s,"%tools.read_csv_tab(input_i,col,line))
                        else:
                            if col == 0: # replace class names by an integer
                                file.write("%i,"%class_number)
                            
                            else:
                                if col == 1:
                                    laurent=1
                                    onto.write("%i,%s,%i,%s\n"%(Iteration,i,class_number,tools.read_csv_tab(input_i,col,line)))
                                    Iteration=Iteration+1
                                else:
                                    file.write("%s,"%tools.read_csv_tab(input_i,col,line))
                    file.write("\n")
                file.close()
                class_number = class_number + 1
                
                if lines > 3 :
                    input_file=Coma_Features+i
                    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1)) # ncol-1 because we skip the class names
                    X = data[:, :ncol]
                    y = data[:, 0].astype(np.int) # Labels (class)
                    
                    #Replace missing values 'nan' by column mean
                    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
                    imp.fit(X)
                    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
                    # Output replacement "Nan" values
                    Y=imp.transform(X)
                    
                    #Data Standardization
                    if Norm == 'normalize':
                        Z=preprocessing.normalize(Y, axis=0, norm='l2') # Normalize
                        #print Z
                    else:
                        if Norm == 'binarize':
                            binarizer=preprocessing.Binarizer().fit(Y) # Binarize for Bernoulli
                            Z = binarizer.transform(Y)
                            #print Z
                        else:
                            if Norm == 'standardize':
                                min_max_scaler = preprocessing.MinMaxScaler() # Normalize the data to [0,1]
                                Z=min_max_scaler.fit_transform(Y)
                                #print Z
                            else:
                                Z=preprocessing.scale(Y) #Scaling
                    
                    #Create new files with corrected and standardized data
                    output_file=Corrected_Features+i
                    file = open(output_file, "w")
                    writer=csv.writer(file, lineterminator=',')
                    feature=[26,48,72]
                    for line_1 in xrange(lines-1):
                        for col_1 in xrange(ncol-1):
                            if col_1==0:
                                file.write("%s,"%y[line_1])
                            else:
                                if col_1 in feature:
                                    file.write("%f,"%Z[line_1,col_1])
                        file.write("\n")
                    file.close()
                else:
                    laurent=1
    onto.close()


#################################################################################################

#                                      file merger                                              #

#################################################################################################

# Without Labels
# Merge all the files from corrected files to one single file skipping titles
def merger(Preprocessed_file,file_name,Corrected_Features):
    # Merge all features files into one
    lvltrace.lvltrace("LVLEntree dans merger data_preproc")
    fout=open(Preprocessed_file,"a")
    # first file:
    first_file=Corrected_Features+file_name
    for line in open(first_file):
        fout.write(line)
    # now the rest:
    for root, dirs, files in os.walk(Corrected_Features):
        for i in files:
            if i != file_name:
                if not i.startswith('.'):
                    f=open(Corrected_Features+i)
                    #f.next() #Skip the header
                    for line in f:
                        fout.write(line)
                    f.close()
    fout.close()
    lvltrace.lvltrace("LVLSortie dans merger data_preproc")

# With Labels
# Merge all the files from corrected files to one single file skipping titles
def merger_labelled(Preprocessed_file,file_name,Corrected_Features):
    lvltrace.lvltrace("LVLEntree dans merger_labelled dabs data_preproc")
    # Merge all features files into one
    file_random=Corrected_Features+'/'+file_name
    ncol=tools.file_col_coma(file_random)
    data = np.loadtxt(file_random, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    n_samples, n_features = X.shape
    fout=open(Preprocessed_file,"a")
    fout.write("Class,")
    for n in xrange(1,n_features+1):
        fout.write("feature_%i,"%n)
    fout.write("\n")
    # first file:
    first_file=Corrected_Features+file_name
    for line in open(first_file):
        fout.write(line)
    # now the rest:
    for root, dirs, files in os.walk(Corrected_Features):
        for i in files:
            if i != file_name:
                if not i.startswith('.'):
                    f=open(Corrected_Features+i)
                    f.next() #Skip the header
                    for line in f:
                        fout.write(line)
                    f.close()
    fout.close()
    lvltrace.lvltrace("LVLSortie dans merger_labelled dans data_preproc")

#################################################################################################

#                                      Similiraity Matrix                                       #

#################################################################################################

#create a similarity_matrix without labels
def similarity_matrix(Corrected_Features,similarity):
    file = open(similarity, "w")
    writer=csv.writer(file, lineterminator=',')
    for root, dirs, files in os.walk(Corrected_Features):
        for j in files:
            if not j.startswith('.'):
                input_1=Corrected_Features+j
                ncol1=tools.file_col_coma(input_1)-1
                data = np.loadtxt(input_1, delimiter=',', usecols=range(ncol1)) # ncol-1 because we skip the neuron manes
                X1 = data[:,:]
                y1 = data[:, 0].astype(np.int) # Labels (class)
                mtype1=[0 for x in xrange(ncol1-1)]
                f=[0 for x in xrange(ncol1-1)]
                for col in xrange(1,ncol1):
                    mtype1[col-1]=np.mean(X1[:,col]) #mean of each feature
                for root, dirs, files in os.walk(Corrected_Features):
                    for i in files:
                        if not i.startswith('.'):
                            input_2=Corrected_Features+i
                            ncol2=tools.file_col_coma(input_2)-1
                            data = np.loadtxt(input_2, delimiter=',', usecols=range(ncol2)) # ncol-1 because we skip the neuron names
                            X2 = data[:,:]
                            y2 = data[:, 0].astype(np.int) # Labels (class)
                            mtype2=[0 for x in xrange(ncol2-1)]
                            for col in xrange(1,ncol2):
                                mtype2[col-1]=np.mean(X2[:,col]) #mean of each feature
                            for col in xrange(1,ncol2):
                                f[col-1]=np.abs(mtype2[col-1]-mtype1[col-1])
                            similarity=np.mean(f)
                            file.write("%f,"%similarity)
                file.write("\n")
    file.close()

#create a similarity_matrix with labels
def neuron_similarity_matrix_labelled(Preprocessed_file,similarity,Corrected_Features):
    lvltrace.lvltrace("LVLEntree dans neuron_similarity_matrix_labelled data_preproc")
    file = open(similarity, "w")
    writer=csv.writer(file, lineterminator=',')
    file.write("mtype,")
    for root, dirs, files in os.walk(Corrected_Features):
        for v in files:
            if not v.startswith('.'):
                input=Corrected_Features+v
                ncol=tools.file_col_coma(input)-1
                data = np.loadtxt(input, delimiter=',', usecols=range(ncol)) # ncol-1 because we skip the neuron manes
                y = data[:, 0].astype(np.int) # Labels (class)
                file.write("%i,"%np.mean(y))
        file.write("\n")
    for root, dirs, files in os.walk(Corrected_Features):
        for j in files:
            if not j.startswith('.'):
                input_1=Corrected_Features+j
                ncol1=tools.file_col_coma(input_1)-1
                data = np.loadtxt(input_1, delimiter=',', usecols=range(ncol1)) # ncol-1 because we skip the neuron manes
                X1 = data[:,:]
                y1 = data[:, 0].astype(np.int) # Labels (class)
                mtype1=[0 for x in xrange(ncol1-1)]
                f=[0 for x in xrange(ncol1-1)]
                label1=np.mean(y1)
                for col in xrange(1,ncol1):
                    mtype1[col-1]=np.mean(X1[:,col]) #mean of each feature
                file.write("%i,"%label1)
                for root, dirs, files in os.walk(Corrected_Features):
                    for i in files:
                        if not i.startswith('.'):
                            input_2=Corrected_Features+i
                            ncol2=tools.file_col_coma(input_2)-1
                            data = np.loadtxt(input_2, delimiter=',', usecols=range(ncol2)) # ncol-1 because we skip the neuron manes
                            X2 = data[:,:]
                            y2 = data[:, 0].astype(np.int) # Labels (class)
                            mtype2=[0 for x in xrange(ncol2-1)]
                            
                            for col in xrange(1,ncol2):
                                mtype2[col-1]=np.mean(X2[:,col]) #mean of each feature
                            
                            for col in xrange(1,ncol2):
                                f[col-1]=np.abs(mtype2[col-1]-mtype1[col-1])
                            similarity=np.mean(f)
                            file.write("%f,"%similarity)
                file.write("\n")
    file.close()
    lvltrace.lvltrace("LVLSortie dans neuron_similarity_matrix_labelled data_preproc")
