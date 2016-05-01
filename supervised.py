# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

#######################################################################################
#                                                                                     #
#                    SUPERVISED CLASSIFIERS: Dependencies                             #
#                                                                                     #
#######################################################################################

import tools
import inputs
import lvltrace
import configuration
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import ExtraTreesClassifier
import pylab as pl
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn import tree
from sklearn.cross_validation import train_test_split



#################################################################################################

#                                       SUPERVISED ALGORITHMS                                   #

#################################################################################################


def plot_confusion_matrix(y, y_pred,title,save):
   plt.imshow(metrics.confusion_matrix(y, y_pred),
               cmap=plt.cm.binary, interpolation='nearest')
   plt.colorbar()
   plt.xlabel('True Value')
   plt.ylabel('Predicted Value')
   plt.title(title)
   plt.savefig(save)
   plt.close()

# Gaussian Naive Bayes estimator
def gaussianNB(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans gaussianNB")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    # Instantiate the estimator
    clf = GaussianNB()
    # Fit the estimator to the data
    clf.fit(X, y)
    # Use the model to predict the last several labels
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "Gaussian Naive Bayes estimator accuracy"
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"GaussianNB_metrics.txt"
    file = open(results, "w")
    file.write("Gaussian Naive Bayes estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "Gaussian Naive Bayes"
    save = Output + "Gaussian_NB_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans gaussianNB")


# Multinomial Naive Bayes estimator
def multinomialNB(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans multinomialNB")
    try:
        ncol=tools.file_col_coma(input_file)
        data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
        X = data[:,1:]
        y = data[:,0]
        n_samples, n_features = X.shape
        # Instantiate the estimator
        clf = MultinomialNB()
        # Fit the estimator to the data
        clf.fit(X, y)
        # Use the model to predict the last several labels
        y_pred = clf.predict(X)
        print "#########################################################################################################\n"
        print "Multinomial Naive Bayes estimator accuracy "
        print "classification accuracy:", metrics.accuracy_score(y, y_pred)
        print "precision:", metrics.precision_score(y, y_pred)
        print "recall:", metrics.recall_score(y, y_pred)
        print "f1 score:", metrics.f1_score(y, y_pred)
        print "\n"
        print "#########################################################################################################\n"
        results = Output+"Multinomial_NB_metrics.txt"
        file = open(results, "w")
        file.write("Multinomial Naive Bayes estimator accuracy\n")
        file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
        file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
        file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
        file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
        file.write("True Value, Predicted Value, Iteration\n")
        for n in xrange(len(y)):
            file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
        file.close()
        title = "Multinomial Naive Bayes"
        save = Output + "Multinomial_NB_confusion_matrix.png"
        plot_confusion_matrix(y, y_pred,title,save)
    except (ValueError):
        if configuration.normalization == 'normalize':
            results = Output+"Multinomial_NB_metrics.txt"
            file = open(results, "w")
            file.write("In configuration.py file, normalization=normalize -- Input Values must be superior to 0\n")
            file.close()
    lvltrace.lvltrace("LVLSortie dans multinomialNB")


# KNeighbors
def kneighbors(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans kneighbors")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "KNeighbors Accuracy "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"KNeighbors_metrics.txt"
    file = open(results, "w")
    file.write("KNeighbors estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "KNeighbors"
    save = Output + "KNeighbors_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans kneighbors")

# Radius Neighbors
def Radius_Neighbors(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans Radius_Neighbors")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf = RadiusNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "Radius Neighbors Accuracy "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"Raidus_Neighbors_metrics.txt"
    file = open(results, "w")
    file.write("Radius Neighbors estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "Radius Neighbors"
    save = Output + "Radius_Neighbors_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans Radius_Neighbors")

# Linear Discriminant Analaysis
def lda(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans lda")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    #lda=LDA(n_components=2)
    lda=LDA()
    lda.fit(X,y)
    X_LDA = lda.transform(X)
    y_pred = lda.predict(X)
    print "#########################################################################################################\n"
    print "Linear Discriminant Analysis Accuracy "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"LDA_metrics.txt"
    file = open(results, "w")
    file.write("Linear Discriminant Analaysis estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "LDA"
    save = Output + "LDA_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    # plot the results along with the labels
    fig, ax = plt.subplots()
    im = ax.scatter(X_LDA[:, 0], X_LDA[:, 1], c=y)
    fig.colorbar(im);
    save_lda = Output + "LDA_plot.png"
    plt.savefig(save_lda)
    plt.close()
    lvltrace.lvltrace("LVLSortie dans lda")

# Quadratic Discriminant Analaysis
def qda(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans qda")
    try:
        ncol=tools.file_col_coma(input_file)
        data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
        X = data[:,1:]
        y = data[:,0]
        n_samples, n_features = X.shape
        qda=QDA()
        qda.fit(X,y)
        y_pred = qda.predict(X)
        print "#########################################################################################################\n"
        print "Quadratic Discriminant Analysis Accuracy "
        print "classification accuracy:", metrics.accuracy_score(y, y_pred)
        print "precision:", metrics.precision_score(y, y_pred)
        print "recall:", metrics.recall_score(y, y_pred)
        print "f1 score:", metrics.f1_score(y, y_pred)
        print "\n"
        print "#########################################################################################################\n"
        results = Output+"QDA_metrics.txt"
        file = open(results, "w")
        file.write("Quadratic Discriminant Analaysis estimator accuracy\n")
        file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
        file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
        file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
        file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
        file.write("\n")
        file.write("True Value, Predicted Value, Iteration\n")
        for n in xrange(len(y)):
            file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
        file.close()
        title = "QDA"
        save = Output + "QDA_confusion_matrix.png"
        plot_confusion_matrix(y, y_pred,title,save)
    except (AttributeError):
        if configuration.normalization == 'normalize':
            results = Output+"Multinomial_NB_metrics.txt"
            file = open(results, "w")
            file.write("In configuration.py file, normalization='normalize' -- Input Values must be superior to 0\n")
            file.close()
    lvltrace.lvltrace("LVLSortie dans qda")

# Support Vector Machine
# C-Support Vector Classifcation (with RBF kernel)
def SVC_rbf(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans SVC_rbf")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf=svm.SVC(kernel='linear')
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "C-Support Vector Classifcation (with RBF kernel) "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"SVM_RBF_Kernel_metrics.txt"
    file = open(results, "w")
    file.write("Support Vector Machine with RBF Kernel estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "SVC - RBF Kernel"
    save = Output + "SVC_RBF_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans SVC_rbf")

# C-Support Vector Classifcation (with linear kernel)
def SVC_linear(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans SVC_linear")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf=svm.SVC(kernel='linear')
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "C-Support Vector Classifcation (with linear kernel) "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"SVM_Linear_Kernel_metrics.txt"
    file = open(results, "w")
    file.write("Support Vector Machine with Linear Kernel estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "SVC - linear Kernel"
    save = Output + "SVC_linear_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans SVC_linear")

# Stochastic Gradient Descent
def stochasticGD(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans stochasticGD")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "Stochastic Gradient Descent "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"Stochastic_GD_metrics.txt"
    file = open(results, "w")
    file.write("Stochastic Gradient Descent estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "Stochastic Gradient Descent"
    save = Output + "Stochastic_GD_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans stochasticGD")

# Nearest Centroid Classifier
def nearest_centroid(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans nearest_centroid")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf = NearestCentroid()
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "Nearest Centroid Classifier "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"Nearest_Centroid_metrics.txt"
    file = open(results, "w")
    file.write("Nearest Centroid Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "Nearest Centroid Classifier"
    save = Output + "Nearest_Centroid_Classifier_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans nearest_centroid")

# Decision Trees
def decisiontreeclassifier(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans decisiontreeclassifier")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "Decision Trees "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"Decision_Trees_metrics.txt"
    file = open(results, "w")
    file.write("Decision Trees Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "Decision Trees"
    save = Output + "Decision_Trees_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans decisiontreeclassifier")

# The Random forest algo
def randomforest(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans randomforest")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print "The Random forest algo "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "f1 score:", metrics.f1_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"Random_Forest_metrics.txt"
    file = open(results, "w")
    file.write("Random Forest Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "The Random forest"
    save = Output + "Random_Forest_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans randomforest")

# Extremely Randomized Trees
def extratreeclassifier(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans extratreeclassifier")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    clf = ExtraTreesClassifier(n_estimators=10)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    print "#########################################################################################################\n"
    print " Extremely Randomized Trees "
    print "classification accuracy:", metrics.accuracy_score(y, y_pred)
    print "precision:", metrics.precision_score(y, y_pred)
    print "recall:", metrics.recall_score(y, y_pred)
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"_Extremely_Random_Forest_metrics.txt"
    file = open(results, "w")
    file.write("Extremely Random Forest Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],y_pred[n],(n+1)))
    file.close()
    title = "Extremely Randomized Trees"
    save = Output + "Extremely_Randomized_Trees_confusion_matrix.png"
    plot_confusion_matrix(y, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans extratreeclassifier")

