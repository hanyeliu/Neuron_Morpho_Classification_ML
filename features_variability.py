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
import matplotlib as mpl
import pandas as pd
from numpy.random import randn
import time
from multiprocessing import Pool
import multiprocessing

def features_variability(descriptive_analysis_output,descriptive_variability,cores):
    lvltrace.lvltrace("LVLEntree dans features_variability ")
    path_csv_analysis=descriptive_analysis_output
    #print path_csv_analysis
    Score=descriptive_variability+"Global_Features_Variability.csv"
    file = open(Score, "w")
    writer=csv.writer(file, lineterminator=',')

    # Create a pool of processes depending on number of CPUs available,
    # with each file in input_files to be loaded using the appropriate function
    cpuCount = multiprocessing.cpu_count()
    #if cpuCount > 12: cpuCount = 12 # Limit processor count to 12 when executing on server
    if cores == None:
        cores = cpuCount
    if cores > 1: s='s'
    else: s=''
    if cores == False: # Sequential loading used for debugging
        for root, dirs, files in os.walk(path_csv_analysis):
            for i in files:
                if not i.startswith('.') and i.endswith('.csv'):
                        scoring=features_variability_single(path_csv_analysis,i)
		        file.write("%s,"%os.path.splitext(i)[0])
    			file.write("%f\n"%scoring)
	file.close()
    else: # Multi-process loading
        pool = Pool(processes=cores)
        data = []
        b=[0 for x in xrange(len(os.listdir(path_csv_analysis)))]
        iter=0
        for root, dirs, files in os.walk(path_csv_analysis):
            for i in files:
                if not i.startswith('.') and i.endswith('.csv'):
                        csv_file=path_csv_analysis+i
                        #print csv_file
                        #print i,iter
                        #b[iter]=os.path.splitext(i)[0]
                        b[iter]=i
                        iter=iter+1
                        #data.append((csv_file,i,file))
                        #print csv_file
                        data.append((path_csv_analysis,i))
        scoring_multi=pool.map(features_variability_multi_cores,data)
	for k in xrange(iter):
		file.write("%s,"%b[k])
                file.write("%f\n"%scoring_multi[k])
	file.close()
    variability_plotting(descriptive_variability,Score)
    lvltrace.lvltrace("LVLSortie dans features_variability ")

def features_variability_multi_cores(data):
    add=0
    scoring=0
    csv_file=data[0]+data[1]
    lines=tools.file_lines_coma(csv_file)
    features = np.genfromtxt(csv_file,delimiter=',',dtype=None,usecols=0)
    #print features
    for j in xrange(lines):
        if tools.read_float_coma(csv_file,2,j) >= 3 :
            if tools.read_float_coma(csv_file,8,j) <= 33.3 :
                add=add+1
            else:
                add=add
        else:
                add=add
    scoring=float((float(add)/(len(features)-1))*100)
    return scoring

def features_variability_single(path_csv_analysis,i):
    add=0
    scoring=0
    csv_file=path_csv_analysis+i
    features = np.genfromtxt(csv_file,delimiter=',',dtype=None,usecols=0)
    #print features
    #w = features[0, :]
    #print w
    lines=tools.file_lines_coma(csv_file)
    for j in xrange(lines):
        if tools.read_float_coma(csv_file,2,j) >= 3 :
            if tools.read_float_coma(csv_file,8,j) <= 33.3 :
                add=add+1
            else:
                add=add
        else:
            add=add
    scoring=float((float(add)/(len(features)-1))*100)
    return scoring


def variability_plotting(descriptive_variability,Score):
    tools.sortcsvbymanyfields(Score,1)
    df = pd.read_csv(Score,index_col=False)
    b = df.ix[:,1]
    a = df.ix[:,0]
    plt.figure()
    y=np.array(a)
    X=np.arange(len(a))
    x=np.array(b)
    plt.barh(X, x, align='center',  height=0.8, color='blue', alpha=.5)
    plt.yticks(X, y)
    plt.tick_params(axis='y', labelsize=7)
    plt.xlabel('Scores in Percentage')
    plt.ylabel('Cell Classes')
    plt.title("Features Variability Scores by Class")
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15, bottom=None, right=0.98, top=0.95, wspace=0, hspace=0)
    plt.grid(color='lightgray', alpha=0.7)
    save_plot=descriptive_variability+"Global_Features_Variability.png"
    plt.savefig(save_plot)
    plt.close()



