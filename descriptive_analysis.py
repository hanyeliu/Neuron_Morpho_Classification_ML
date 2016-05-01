# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

import tools
import lvltrace
import os
import numpy as np
from numpy import sort
import csv
from scipy import stats
#from scipy.stats import scoreatpercentile
import math
#from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# Display
import mpld3
from mpld3.display import display_d3
from IPython.display import display
from IPython.display import HTML
import time
from multiprocessing import Pool
import multiprocessing

#Descriptive analysis for each class by features

def descriptive_analysis(Extracted_Features,descriptive_analysis_output,cores,Corrected_Features_descriptive):
    lvltrace.lvltrace("LVLEntree dans descriptive_analysis")
    Input_population_analysis=Corrected_Features_descriptive
    folder = descriptive_analysis_output
    # Create a pool of processes depending on number of CPUs available,
    # with each file in input_files to be loaded using the appropriate function
    cpuCount = multiprocessing.cpu_count()
    #if cpuCount > 12: cpuCount = 12 # Limit processor count to 12 when executing on server
    if cores == None:
        cores = cpuCount
    if cores > 1: s='s'
    else: s=''
    if cores == False: # Sequential loading used for debugging
        for root, dirs, files in os.walk(Input_population_analysis):
            for i in files:
                if not i.startswith('.'):
                    descriptive_single(Extracted_Features,descriptive_analysis_output,i,Input_population_analysis)

    else: # Multi-process loading
        pool = Pool(processes=cores)
        data = []
        for root, dirs, files in os.walk(Input_population_analysis):
            for i in files:
                if not i.startswith('.'):
                        morphology_csv=Input_population_analysis+i
                        features = np.genfromtxt(morphology_csv,delimiter='\t',dtype=None)
                        w = features[0, :]
                        data.append((Extracted_Features,descriptive_analysis_output,i,morphology_csv,w))
        pool.map(descriptive_multi_cores,data)
    lvltrace.lvltrace("LVLSortie dans descriptive_analysis")

def descriptive_multi_cores(data):
	fig, ax = plt.subplots()
	lines=tools.file_lines(data[3])
	ncol=tools.file_col(data[3])-1
	outputs_files=data[1]+data[2]

	if lines==1:
	    file = open(outputs_files, "w")
	    writer=csv.writer(file, lineterminator=',')
	    b=[[0 for x in xrange(6)] for x in xrange(ncol-2)]
	    normality=[0 for x in xrange(1)]
	    file.write("Variables,Class,N,Mean,Std_Dev,Variance,Max,Min,Coeff_Var,Interquartile,Normality_distrib,Confident_intervals_left,Confident_intervals_right\n")
	    for col in xrange(ncol-2):
		a = tools.read_float_tab(data[3],col+2,1)
		b[col][0]=1
		b[col][1]=float(a)
		b[col][2]=0
		b[col][3]=0
		b[col][4]=float(a)
		b[col][5]=float(a)
		file.write("%s,"%data[4][col+2])
		file.write("%s,"%(os.path.splitext(data[2])[0]))
		file.write("%f,"%b[col][0])
		file.write("%f,"%b[col][1])
		file.write("%f,"%b[col][2])
		file.write("%f,"%b[col][3])
		file.write("%f,"%b[col][4])
		file.write("%f,"%b[col][5])
		file.write("100,")
		file.write("0,")
		file.write("0,")
		file.write("0,")
		file.write("0,\n")
	    file.close()
	else:
	    file = open(outputs_files, "w")
	    writer=csv.writer(file, lineterminator=',')
	    b=[[0 for x in xrange(11)] for x in xrange(ncol-2)]
	    file.write("Variables,Class,N,Mean,Std_Dev,Variance,Max,Min,Coeff_Var,Interquartile,Normality_distrib,Confident_intervals_left,Confident_intervals_right\n")
	    a=[0 for x in xrange(lines)]
	    for col in xrange(ncol-2):
		for j in xrange(lines):
		   a[j]=tools.read_float_tab(data[3],col+2,j)
		   b[col][0]=float(len(a))
		b[col][1]=float(np.mean(a))
		b[col][2]=float(np.std(a, dtype=float))
		b[col][3]=float(np.var(a, dtype=float))
		b[col][4]=float(max(a))
		b[col][5]=float(min(a))
		if np.around(b[col][2],decimals=12) != 0.0:
		    if float(len(a)) < 3:
			b[col][6]=float(abs((np.std(a)/np.mean(a))*100))
			file.write("%s,"%data[4][col+2])
			file.write("%s,"%(os.path.splitext(data[2])[0]))
			file.write("%f,"%b[col][0])
			file.write("%f,"%b[col][1])
			file.write("%f,"%b[col][2])
			file.write("%f,"%b[col][3])
			file.write("%f,"%b[col][4])
			file.write("%f,"%b[col][5])
			file.write("%f,"%b[col][6])
			file.write("0,")
			file.write("0,")
			file.write("0,")
			file.write("0,\n")
		    else:
			b[col][6]=float(abs((np.std(a)/np.mean(a))*100))
			X=sort(a)
			upperQuartile = stats.scoreatpercentile(X,.75)
			lowerQuartile = stats.scoreatpercentile(X,.25)
			IQR = upperQuartile - lowerQuartile
			b[col][7]=float(IQR)
			normality=stats.shapiro(a)
			b[col][8]=float(normality[1])
			b[col][9]=np.mean(a)-1.96*(np.std(a)/math.sqrt(len(a)))
			b[col][10]=np.mean(a)+1.96*(np.std(a)/math.sqrt(len(a)))
			file.write("%s,"%data[4][col+2])
			file.write("%s,"%(os.path.splitext(data[2])[0]))
			file.write("%f,"%b[col][0])
			file.write("%f,"%b[col][1])
			file.write("%f,"%b[col][2])
			file.write("%f,"%b[col][3])
			file.write("%f,"%b[col][4])
			file.write("%f,"%b[col][5])
			file.write("%f,"%b[col][6])
			file.write("%f,"%b[col][7])
			file.write("%f,"%b[col][8])
			file.write("%f,"%b[col][9])
			file.write("%f,\n"%b[col][10])
			if normality[1] >= 0.05:
			    norm_distrib = np.linspace(-150,150,100)
			    ax.set_title('Normality of %s class features'%os.path.splitext(data[2])[0])
			    ax.plot(norm_distrib,mlab.normpdf(norm_distrib,np.mean(a),math.sqrt(np.var(a))),label=data[4][col+2],ms=10, alpha=0.3)
			    ax.legend(loc=2, ncol=1, bbox_to_anchor=(0, 0, 1, 0.7),fancybox=True,shadow=False,fontsize=5)
			    ax.grid(color='lightgray', alpha=0.5)
			    ax.patch.set_facecolor('white')
	    
		else:
		    b[col][6]=0
		    X=sort(a)
		    upperQuartile = stats.scoreatpercentile(X,.75)
		    lowerQuartile = stats.scoreatpercentile(X,.25)
		    IQR = upperQuartile - lowerQuartile
		    b[col][7]=float(IQR)
		    b[col][8]=0
		    b[col][9]=np.mean(a)-1.96*(np.std(a)/math.sqrt(len(a)))
		    b[col][10]=np.mean(a)+1.96*(np.std(a)/math.sqrt(len(a)))
		    file.write("%s,"%data[4][col+2])
		    file.write("%s,"%(os.path.splitext(data[2])[0]))
		    file.write("%f,"%b[col][0])
		    file.write("%f,"%b[col][1])
		    file.write("%f,"%b[col][2])
		    file.write("%f,"%b[col][3])
		    file.write("%f,"%b[col][4])
		    file.write("%f,"%b[col][5])
		    file.write("%f,"%b[col][6])
		    file.write("%f,"%b[col][7])
		    file.write("%f,"%b[col][8])
		    file.write("%f,"%b[col][9])
		    file.write("%f,\n"%b[col][10])
	    file.close()
	    display_d3(fig)
	    html = mpld3.fig_to_d3(fig)
	    html_normality=data[1]+data[2]+".html"
	    normality_display = open(html_normality, "w")
	    normality_display.write("%s"%html)
	    normality_display.close()
	    plt.close()


def descriptive_single(Extracted_Features,descriptive_analysis_output,i,Input_population_analysis):
        fig, ax = plt.subplots()
        morphology_csv = Input_population_analysis+i
        features = np.genfromtxt(morphology_csv,delimiter='\t',dtype=None)
        w = features[0, :] # features names
        lines=tools.file_lines(morphology_csv)
        ncol=tools.file_col(morphology_csv)-1
        test=Extracted_Features+i
        outputs_files=descriptive_analysis_output+i
        if lines==1:
            file = open(outputs_files, "w")
            writer=csv.writer(file, lineterminator=',')
            b=[[0 for x in xrange(6)] for x in xrange(ncol-2)]
            normality=[0 for x in xrange(1)]
            file.write("Variables,Class,N,Mean,Std_Dev,Variance,Max,Min,Coeff_Var,Interquartile,Normality_distrib,Confident_intervals_left,Confident_intervals_right\n")
            for col in xrange(ncol-2):
                a = tools.read_float_tab(morphology_csv,col+2,1)
                b[col][0]=1
                b[col][1]=float(a)
                b[col][2]=0
                b[col][3]=0
                b[col][4]=float(a)
                b[col][5]=float(a)
                file.write("%s,"%w[col+2])
                file.write("%s,"%(os.path.splitext(i)[0]))
                file.write("%f,"%b[col][0])
                file.write("%f,"%b[col][1])
                file.write("%f,"%b[col][2])
                file.write("%f,"%b[col][3])
                file.write("%f,"%b[col][4])
                file.write("%f,"%b[col][5])
                file.write("100,")
                file.write("0,")
                file.write("0,")
                file.write("0,")
                file.write("0,\n")
            file.close()
        else:
            file = open(outputs_files, "w")
            writer=csv.writer(file, lineterminator=',')
            b=[[0 for x in xrange(11)] for x in xrange(ncol-2)]
            file.write("Variables,Class,N,Mean,Std_Dev,Variance,Max,Min,Coeff_Var,Interquartile,Normality_distrib,Confident_intervals_left,Confident_intervals_right\n")
            a=[0 for x in xrange(lines)]
            for col in xrange(ncol-2):
                for j in xrange(lines):
                   a[j]=tools.read_float_tab(morphology_csv,col+2,j)
                   b[col][0]=float(len(a))
                b[col][1]=float(np.mean(a))
                b[col][2]=float(np.std(a, dtype=float))
                b[col][3]=float(np.var(a, dtype=float))
                b[col][4]=float(max(a))
                b[col][5]=float(min(a))
                if np.around(b[col][2],decimals=12) != 0.0:
                    if float(len(a)) < 3:
                        b[col][6]=float(abs((np.std(a)/np.mean(a))*100))
                        file.write("%s,"%w[col+2])
                        file.write("%s,"%(os.path.splitext(i)[0]))
                        file.write("%f,"%b[col][0])
                        file.write("%f,"%b[col][1])
                        file.write("%f,"%b[col][2])
                        file.write("%f,"%b[col][3])
                        file.write("%f,"%b[col][4])
                        file.write("%f,"%b[col][5])
                        file.write("%f,"%b[col][6])
                        file.write("0,")
                        file.write("0,")
                        file.write("0,")
                        file.write("0,\n")
                    else:
                        b[col][6]=float(abs((np.std(a)/np.mean(a))*100))
                        X=sort(a)
                        upperQuartile = stats.scoreatpercentile(X,.75)
                        lowerQuartile = stats.scoreatpercentile(X,.25)
                        IQR = upperQuartile - lowerQuartile
                        b[col][7]=float(IQR)
                        normality=stats.shapiro(a)
                        b[col][8]=float(normality[1])
                        b[col][9]=np.mean(a)-1.96*(np.std(a)/math.sqrt(len(a)))
                        b[col][10]=np.mean(a)+1.96*(np.std(a)/math.sqrt(len(a)))
                        file.write("%s,"%w[col+2])
                        file.write("%s,"%(os.path.splitext(i)[0]))
                        file.write("%f,"%b[col][0])
                        file.write("%f,"%b[col][1])
                        file.write("%f,"%b[col][2])
                        file.write("%f,"%b[col][3])
                        file.write("%f,"%b[col][4])
                        file.write("%f,"%b[col][5])
                        file.write("%f,"%b[col][6])
                        file.write("%f,"%b[col][7])
                        file.write("%f,"%b[col][8])
                        file.write("%f,"%b[col][9])
                        file.write("%f,\n"%b[col][10])
                        if normality[1] >= 0.05:
                            norm_distrib = np.linspace(-150,150,100)
                            ax.set_title('Normality of %s class features'%os.path.splitext(i)[0])
                            ax.plot(norm_distrib,mlab.normpdf(norm_distrib,np.mean(a),math.sqrt(np.var(a))),label=w[col+2],ms=10, alpha=0.3)
                            ax.legend(loc=2, ncol=1, bbox_to_anchor=(0, 0, 1, 0.7),fancybox=True,shadow=False,fontsize=5)
                            ax.grid(color='lightgray', alpha=0.5)
                            ax.patch.set_facecolor('white')
            
                else:
                    b[col][6]=0
                    X=sort(a)
                    upperQuartile = stats.scoreatpercentile(X,.75)
                    lowerQuartile = stats.scoreatpercentile(X,.25)
                    IQR = upperQuartile - lowerQuartile
                    b[col][7]=float(IQR)
                    b[col][8]=0
                    b[col][9]=np.mean(a)-1.96*(np.std(a)/math.sqrt(len(a)))
                    b[col][10]=np.mean(a)+1.96*(np.std(a)/math.sqrt(len(a)))
                    file.write("%s,"%w[col+2])
                    file.write("%s,"%(os.path.splitext(i)[0]))
                    file.write("%f,"%b[col][0])
                    file.write("%f,"%b[col][1])
                    file.write("%f,"%b[col][2])
                    file.write("%f,"%b[col][3])
                    file.write("%f,"%b[col][4])
                    file.write("%f,"%b[col][5])
                    file.write("%f,"%b[col][6])
                    file.write("%f,"%b[col][7])
                    file.write("%f,"%b[col][8])
                    file.write("%f,"%b[col][9])
                    file.write("%f,\n"%b[col][10])
            file.close()
            display_d3(fig)
            html = mpld3.fig_to_d3(fig)
            html_normality=descriptive_analysis_output+i+".html"
            normality_display = open(html_normality, "w")
            normality_display.write("%s"%html)
            normality_display.close()
            plt.close()
