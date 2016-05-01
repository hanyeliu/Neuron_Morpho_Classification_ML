# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

#######################################################################################
#                                                                                     #
#                           SUPERVISED CLASSIFIERS TEST                               #
#                                                                                     #
#######################################################################################
import tools
import lvltrace
import inputs
import time
import configuration
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.cross_validation import train_test_split

#################################################################################################

#                                    SUPERVIZED ALGORITHMS                                      #

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
def gaussianNB(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans gaussianNB split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    # Instantiate the estimator
    clf = GaussianNB()
    # Fit the estimator to the data
    clf.fit(X_train, y_train)
    # Use the model to predict the last several labels
    y_pred = clf.predict(X_test)
    print "Gaussian Naive Bayes estimator accuracy "
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    results = Output+"GaussianNB_metrics_test.txt"
    file = open(results, "w")
    file.write("Gaussian Naive Bayes estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "Gaussian Naive Bayes %f"%test_size
    save = Output + "Gaussian_NB_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLsortie dans gaussianNB split_test")


# Multinomial Naive Bayes estimator
def multinomialNB(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans multinomialNB split_test")
    try:
        ncol=tools.file_col_coma(input_file)
        data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
        X = data[:,1:]
        y = data[:,0]
        n_samples, n_features = X.shape
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        print X_train.shape, X_test.shape
        # Instantiate the estimator
        clf = MultinomialNB()
        # Fit the estimator to the data
        clf.fit(X_train, y_train)
        # Use the model to predict the last several labels
        y_pred = clf.predict(X_test)
        print "Multinomial Naive Bayes estimator accuracy "
        print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
        print "precision:", metrics.precision_score(y_test, y_pred)
        print "recall:", metrics.recall_score(y_test, y_pred)
        print "f1 score:", metrics.f1_score(y_test, y_pred)
        print "\n"
        results = Output+"Multinomial_NB_metrics_test.txt"
        file = open(results, "w")
        file.write("Multinomial Naive Bayes estimator accuracy\n")
        file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
        file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
        file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
        file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
        file.write("True Value, Predicted Value, Iteration\n")
        for n in xrange(len(y_test)):
            file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
        file.close()
        title = "Multinomial Naive Bayes %f"%test_size
        save = Output + "Multinomial_NB_confusion_matrix"+"_%s.png"%test_size
        plot_confusion_matrix(y_test, y_pred,title,save)
    except (ValueError):
        if configuration.normalization == 'normalize':
            results = Output+"Multinomial_NB_metrics_test.txt"
            file = open(results, "w")
            file.write("In configuration.py file, normalization=normalize -- Input Values must be superior to 0\n")
            file.close()
    lvltrace.lvltrace("LVLEntree dans multinomialNB split_test")

# KNeighbors
def kneighbors(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans kneighbors split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print "KNeighbors Accuracy "
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"KNeighbors_metrics_test.txt"
    file = open(results, "w")
    file.write("KNeighbors estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "KNeighbors %f"%test_size
    save = Output + "KNeighbors_confusion_metrics"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans kneighbors split_test")

# Radius Neighbors
def Radius_Neighbors(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans radius_kneighbors split_test")
    try:
        ncol=tools.file_col_coma(input_file)
        data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
        X = data[:,1:]
        y = data[:,0]
        n_samples, n_features = X.shape
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        print X_train.shape, X_test.shape
        clf = RadiusNeighborsClassifier(radius=0.001, weights='uniform', algorithm='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print "Radius Neighbors accuracy "
        print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
        print "precision:", metrics.precision_score(y_test, y_pred)
        print "recall:", metrics.recall_score(y_test, y_pred)
        print "f1 score:", metrics.f1_score(y_test, y_pred)
        print "\n"
        results = Output+"Raidus_Neighbors_metrics_test.txt"
        file = open(results, "w")
        file.write("Radius Neighbors estimator accuracy\n")
        file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
        file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
        file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
        file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
        file.write("\n")
        file.write("True Value, Predicted Value, Iteration\n")
        for n in xrange(len(y_test)):
            file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
        file.close()
        title = "Radius Neighbors %f"%test_size
        save = Output + "Radius_Neighbors_confusion_matrix"+"_%s.png"%test_size
        plot_confusion_matrix(y_test, y_pred,title,save)
    except (ValueError):
        results = Output+"Raidus_Neighbors_metrics_test.txt"
        file = open(results, "w")
        file.write("In configuration.py file:  No neighbors found for test samples, you can try using larger radius, give a label for outliers, consider or removing them from your dataset.")
        file.close()
    lvltrace.lvltrace("LVLSortie dans radius_kneighbors split_test")

# Linear discriminant analysis
def lda(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans lda split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    lda=LDA(n_components=2)
    lda.fit(X_train,y_train)
    X_LDA = lda.transform(X_train)
    print "shape of result:", X_LDA.shape
    y_pred = lda.predict(X_test)
    print "Linear Discriminant Analysis Accuracy "
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    #LVLprint "\n"
    results = Output+"LDA_metrics_test.txt"
    file = open(results, "w")
    file.write("Linear Discriminant Analaysis estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "LDA %f"%test_size
    save = Output + "LDA_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    # plot the results along with the labels
    fig, ax = plt.subplots()
    im = ax.scatter(X_LDA[:, 0], X_LDA[:, 1], c=y_train)
    fig.colorbar(im);
    save_lda = Output + "LDA_plot_test"+"_%s.png"%test_size
    plt.savefig(save_lda)
    plt.close()
    lvltrace.lvltrace("LVLSortie dans lda split_test")

# Quadratic discriminant analysis
def qda(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans qda split_test")
    try:
        ncol=tools.file_col_coma(input_file)
        data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
        X = data[:,1:]
        y = data[:,0]
        n_samples, n_features = X.shape
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        print X_train.shape, X_test.shape
        lda=QDA()
        lda.fit(X_train,y_train)
        y_pred = lda.predict(X_test)
        print "Quadratic Discriminant Analysis Accuracy "
        print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
        print "precision:", metrics.precision_score(y_test, y_pred)
        print "recall:", metrics.recall_score(y_test, y_pred)
        print "f1 score:", metrics.f1_score(y_test, y_pred)
        #LVLprint "\n"
        results = Output+"QDA_metrics_test.txt"
        file = open(results, "w")
        file.write("Quadratic Discriminant Analaysis estimator accuracy\n")
        file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
        file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
        file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
        file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
        file.write("\n")
        file.write("True Value, Predicted Value, Iteration\n")
        for n in xrange(len(y_test)):
            file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
        file.close()
        title = "QDA %f"%test_size
        save = Output + "QDA_confusion_matrix"+"_%s.png"%test_size
        plot_confusion_matrix(y_test, y_pred,title,save)
    except (AttributeError):
        if configuration.normalization == 'normalize':
            results = Output+"Multinomial_NB_metrics_test.txt"
            file = open(results, "w")
            file.write("In configuration.py file, normalization='normalize' -- Input Values must be superior to 0\n")
            file.close()
    lvltrace.lvltrace("LVLSortie dans qda split_test")

# Support Vector Machine
# C-Support Vector Classifcation (with RBF kernel)
def SVC_rbf(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans SVC_rbf split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf=svm.SVC()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "C-Support Vector Classifcation (with RBF kernel) "
    print "y_test, y_pred, iteration"
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"SVM_RBF_Kernel_metrics_test.txt"
    file = open(results, "w")
    file.write("Support Vector Machine with RBF Kernel estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "SVC RBF %f"%test_size
    save = Output + "SVC_rbf_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans SVC_rbf split_test")

# C-Support Vector Classifcation (with RBF linear)
def SVC_linear(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans SVC_linear split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf=svm.SVC(kernel='linear')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "C-Support Vector Classifcation (with RBF linear) "
    print "y_test, y_pred, iteration"
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"SVM_Linear_Kernel_metrics_test.txt"
    file = open(results, "w")
    file.write("Support Vector Machine with Linear Kernel estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "SVC linear %f"%test_size
    save = Output + "SVC_linear_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLsortie dans SVC_linear split_test")

# Stochastic Gradient Descent
def stochasticGD(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans stochasticGD split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "Stochastic Gradient Descent "
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"Stochastic_GD_metrics_test.txt"
    file = open(results, "w")
    file.write("Stochastic Gradient Descent estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "Stochastic Gradient Descent %f"%test_size
    save = Output + "Stochastic_GD_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)

# Nearest Centroid Classifier
def nearest_centroid(input_file,Output,test_size):
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf = NearestCentroid()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "Nearest Centroid Classifier "
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"Nearest_Centroid_metrics_test.txt"
    file = open(results, "w")
    file.write("Nearest Centroid Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "Nearest Centroid %f"%test_size
    save = Output + "Nearest_Centroid_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans stochasticGD split_test")

# Decision Trees
def decisiontreeclassifier(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans decisiontreeclassifier split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "Decision Trees "
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"Decision_Trees_metrics_test.txt"
    file = open(results, "w")
    file.write("Decision Trees Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "Decision Trees %f"%test_size
    save = Output + "Decision_Trees_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans decisiontreeclassifier split_test")

# The Random forest algo
def randomforest(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans randomforest split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "The Random forest"
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"Random_Forest_metrics_test.txt"
    file = open(results, "w")
    file.write("Random Forest Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "Random Forest %f"%test_size
    save = Output + "Random_Forest_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans randomforest split_test")

# Extremely Randomized Trees
def extratreeclassifier(input_file,Output,test_size):
    lvltrace.lvltrace("LVLEntree dans extratreeclassifier split_test")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print X_train.shape, X_test.shape
    clf = ExtraTreesClassifier(n_estimators=10)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "Extremely Randomized Trees"
    print "classification accuracy:", metrics.accuracy_score(y_test, y_pred)
    print "precision:", metrics.precision_score(y_test, y_pred)
    print "recall:", metrics.recall_score(y_test, y_pred)
    print "f1 score:", metrics.f1_score(y_test, y_pred)
    print "\n"
    results = Output+"_Extremely_Random_Forest_metrics_test.txt"
    file = open(results, "w")
    file.write("Extremely Random Forest Classifier estimator accuracy\n")
    file.write("Classification Accuracy Score: %f\n"%metrics.accuracy_score(y_test, y_pred))
    file.write("Precision Score: %f\n"%metrics.precision_score(y_test, y_pred))
    file.write("Recall Score: %f\n"%metrics.recall_score(y_test, y_pred))
    file.write("F1 Score: %f\n"%metrics.f1_score(y_test, y_pred))
    file.write("\n")
    file.write("True Value, Predicted Value, Iteration\n")
    for n in xrange(len(y_test)):
        file.write("%f,%f,%i\n"%(y_test[n],y_pred[n],(n+1)))
    file.close()
    title = "Extremely Randomized Trees %f"%test_size
    save = Output + "Extremely_Randomized_Trees_confusion_matrix"+"_%s.png"%test_size
    plot_confusion_matrix(y_test, y_pred,title,save)
    lvltrace.lvltrace("LVLSortie dans extratreeclassifier split_test")
