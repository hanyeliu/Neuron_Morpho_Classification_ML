# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

import numpy as np
import lvltrace
import pylab as pl
import tools
import csv
import sys
import subprocess
from statsmodels.iolib.foreign import savetxt
from statsmodels.iolib.smpickle import save_pickle
from statsmodels.formula.api import logit, glm
from statsmodels.iolib.summary import Summary
#import feature_names as fn
import os
import pandas as pd

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

##################################### Input #####################################################
#input_file='/Users/vasques/Desktop/neuron_nmo/Results/Selection/Merged_files/features.csv'
#number_of_features = 73
#################################################################################################

def univariate(input_file, Output, percentile):
    ###############################################################################
    # import some data to play with
    lvltrace.lvltrace("LVLEntree dans univariate dans feature_selection")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    sample_size, n_features = X.shape

    ###############################################################################
    pl.figure(1)
    pl.clf()

    X_indices = np.arange(X.shape[-1])
    #print X_indices

    ###############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
    selector = SelectPercentile(f_classif, percentile=percentile)
    selector.fit(X, y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    pl.bar(X_indices - .45, scores, width=.2,
           label=r'Univariate score ($-Log(p_{value})$)', color='g')

    ###############################################################################
    # Compare to the weights of an SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= svm_weights.max()

    pl.bar((X_indices+1) - .25, svm_weights, width=.2, label='SVM weight', color='r')

    clf_selected = svm.SVC(kernel='linear')
    clf_selected.fit(selector.transform(X), y)

    svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.max()

    pl.bar(X_indices[selector.get_support()] - .05, svm_weights_selected, width=.2,
           label='SVM weights after selection', color='b')
           
    pl.title("Feature selection")
    pl.xlabel('Feature number')
    pl.yticks(())
    pl.axis('tight')
    pl.legend(loc='upper right')
    save=Output+"univariate.png"
    pl.savefig(save)
    # Print the feature ranking
    results = Output+"univariate.txt"
    file = open(results, "w")
    file.write("Feature Ranking\n")
    #print len(X_indices[selector.get_support()])
    for i in xrange(len(X_indices[selector.get_support()])):
        #print i
        #print (X_indices[selector.get_support()][i]+1)
        #print svm_weights_selected[i]
        file.write("%f,%f\n"%((X_indices[selector.get_support()][i]+1),svm_weights_selected[i]))
    file.close()
    
    #print("Feature ranking:")
    #print (X_indices[selector.get_support()] +1)
    #print svm_weights_selected
    lvltrace.lvltrace("LVLSortie dans univariate dans feature_selection")


def forest_of_trees(input_file,Output):

    import numpy as np

    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier
    lvltrace.lvltrace("LVLEntree dans forest_of_trees dans feature_selection")

    # Build a classification task using 3 informative features
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    #print X
    #print y
    sample_size, n_features = X.shape

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    results = Output+"forest_of_tree.txt"
    file = open(results, "w")
    file.write("Feature Ranking\n")
    
    # Print the feature ranking
    #print("Feature ranking:")

    for f in range(n_features):
        file.write("%d. feature %d (%f)\n" % (f + 1, indices[f]+1, importances[indices[f]]))
        #print("%d. feature %d (%f)" % (f + 1, indices[f]+1, importances[indices[f]]))
    file.close()
    # Plot the feature importances of the forest
    import pylab as pl
    pl.figure()
    pl.title("Feature importances: Forest of trees applied to Layers + Types")
    pl.bar(range(n_features), importances[indices],
           color="r", yerr=std[indices], align="center")
    pl.xticks(range(n_features), indices+1)
    pl.axis('tight')
    #pl.xlim([-1, 73])
    save=Output+"forest_of_tree.png"
    pl.savefig(save)
    lvltrace.lvltrace("LVLSortie dans forest_of_trees dans feature_selection")

def predict_class_glm(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans predict_class_glm dans feature_selection")
    csv=input_file
    df = pd.read_csv(csv)
    #print df
    df = df[['Class','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','feature_9','feature_10','feature_11','feature_12','feature_13','feature_14','feature_15','feature_16','feature_17','feature_18','feature_19','feature_20','feature_21','feature_22','feature_23','feature_24','feature_25','feature_26','feature_27','feature_28','feature_29','feature_30','feature_31','feature_32','feature_33','feature_34','feature_35','feature_36','feature_37','feature_38','feature_39','feature_40','feature_41','feature_42','feature_43']].dropna()
    df.head()
    logit = glm(formula='Class  ~  feature_1+feature_2+feature_3+feature_4+feature_5+feature_6+feature_7+feature_8+feature_9+feature_10+feature_11+feature_12+feature_13+feature_14+feature_15+feature_16+feature_17+feature_18+feature_19+feature_20+feature_21+feature_22+feature_23+feature_24+feature_25+feature_26+feature_27+feature_28+feature_29+feature_30+feature_31+feature_32+feature_33+feature_34+feature_35+feature_36+feature_37+feature_38+feature_39+feature_40+feature_41+feature_42+feature_43', data=df).fit()
    print logit.summary()
    save = Output + "glm.txt"
    old_stdout = sys.stdout
    log_file = open(save,"w")
    sys.stdout = log_file
    print logit.summary()
    sys.stdout = old_stdout
    log_file.close()
    lvltrace.lvltrace("LVLSortie dans predict_class_glm dans feature_selection")


