# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

#  unsupervised.py
#  unsupervised classification functions
#
#
#  Updated by Xavier Vasques on 19/02/2015.
#

#################################################################################################
#                                                                                               #
#                                  UNSUPERVISED CLASSIFIERS                                     #
#                                                                                               #
#################################################################################################
import tools
import lvltrace
import configuration
import input_files
import inputs
import numpy as np
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pylab as pl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from time import time
import numpy as np
import pylab as pl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
import time as time
import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3

##################################### C L U S T E R I N G #######################################

############################################ KMEANS #############################################
def kmeans(input_file, n_clusters, Output):
    lvltrace.lvltrace("LVLEntree dans kmeans unsupervised")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    sample_size, n_features = X.shape
    k_means=cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(X)
    reduced_data = k_means.transform(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    print "#########################################################################################################\n"
    #print y
    #print labels
    print "K-MEANS\n"
    print('homogeneity_score: %f'%metrics.homogeneity_score(y, labels))
    print('completeness_score: %f'%metrics.completeness_score(y, labels))
    print('v_measure_score: %f'%metrics.v_measure_score(y, labels))
    print('adjusted_rand_score: %f'%metrics.adjusted_rand_score(y, labels))
    print('adjusted_mutual_info_score: %f'%metrics.adjusted_mutual_info_score(y,  labels))
    print('silhouette_score: %f'%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    print('\n')
    print "#########################################################################################################\n"
    results = Output+"kmeans_scores.txt"
    file = open(results, "w")
    file.write("K-Means Scores\n")
    file.write("Homogeneity Score: %f\n"%metrics.homogeneity_score(y, labels))
    file.write("Completeness Score: %f\n"%metrics.completeness_score(y, labels))
    file.write("V-Measure: %f\n"%metrics.v_measure_score(y, labels))
    file.write("The adjusted Rand index: %f\n"%metrics.adjusted_rand_score(y, labels))
    file.write("Adjusted Mutual Information: %f\n"%metrics.adjusted_mutual_info_score(y,  labels))
    file.write("Silhouette Score: %f\n"%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    file.write("\n")
    file.write("True Value, Cluster numbers, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f, %f, %i\n"%(y[n],labels[n],(n+1)))
    file.close()
    import pylab as pl
    from itertools import cycle
    # plot the results along with the labels
    k_means_cluster_centers = k_means.cluster_centers_
    fig, ax = plt.subplots()
    im=ax.scatter(X[:, 0], X[:, 1], c=labels, marker='.')
    for k in xrange(n_clusters):
        my_members = labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(cluster_center[0], cluster_center[1], 'w', color='b',
                marker='x', markersize=6)
    fig.colorbar(im)
    plt.title("Number of clusters: %i"%n_clusters)
    save = Output + "kmeans.png"
    plt.savefig(save)
    lvltrace.lvltrace("LVLsortie dans kmeans unsupervised")


############################################# MiniBatchKMeans #############################################
def MiniBatchKMeans(input_file, n_clusters, Output):
    lvltrace.lvltrace("LVLEntree dans MiniBatchKMeans unsupervised")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    sample_size, n_features = X.shape
    k_means=cluster.MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(X)
    # y_pred = k_means.predict(X) # same as labels
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    print "#########################################################################################################\n"
    print "Mini Batch K-MEANS"
    #print labels
    #print y
    print('homogeneity_score: %f'%metrics.homogeneity_score(y, labels))
    print('completeness_score: %f'%metrics.completeness_score(y, labels))
    print('v_measure_score: %f'%metrics.v_measure_score(y, labels))
    print('adjusted_rand_score: %f'%metrics.adjusted_rand_score(y, labels))
    print('adjusted_mutual_info_score: %f'%metrics.adjusted_mutual_info_score(y,  labels))
    print('silhouette_score: %f'%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    print('\n')
    print "#########################################################################################################\n"
    results = Output+"kmeans_metrics.txt"
    file = open(results, "w")
    file.write("K-Means\n")
    file.write("Homogeneity Score: %f\n"%metrics.homogeneity_score(y, labels))
    file.write("Completeness Score: %f\n"%metrics.completeness_score(y, labels))
    file.write("V-Measure: %f\n"%metrics.v_measure_score(y, labels))
    file.write("The adjusted Rand index: %f\n"%metrics.adjusted_rand_score(y, labels))
    file.write("Adjusted Mutual Information: %f\n"%metrics.adjusted_mutual_info_score(y,  labels))
    file.write("Silhouette Score: %f\n"%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    file.write("\n")
    file.write("True Value, Clusters, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],labels[n],(n+1)))
    file.close()
    # plot the results along with the labels
    k_means_cluster_centers = k_means.cluster_centers_
    fig, ax = plt.subplots()
    im=ax.scatter(X[:, 0], X[:, 1], c=labels, marker='.')
    for k in xrange(n_clusters):
        my_members = labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(cluster_center[0], cluster_center[1], 'w', color='b',
                marker='x', markersize=6)
    fig.colorbar(im);
    plt.title("Number of clusters: %i"%n_clusters)
    save = Output + "mini_batch_kmeans.png"
    plt.savefig(save)
    lvltrace.lvltrace("LVLsortie dans MiniBatchKMeans unsupervised")

############################################# K-Means clustering PCA-reduced data #############################################
def KMeans_PCA(input_file, n_clusters, Output):
    lvltrace.lvltrace("LVLEntree dans KMeans_PCA unsupervised")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    sample_size, n_features = X.shape
    reduced_data = PCA(n_components=2).fit_transform(X)
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=50)
    k_means.fit(reduced_data)
    labels = k_means.labels_
    print "#########################################################################################################\n"
    print "K-MEANS on PCA-reduced data"
    #print labels
    #print y
    print('homogeneity_score: %f'%metrics.homogeneity_score(y, labels))
    print('completeness_score: %f'%metrics.completeness_score(y, labels))
    print('v_measure_score: %f'%metrics.v_measure_score(y, labels))
    print('adjusted_rand_score: %f'%metrics.adjusted_rand_score(y, labels))
    print('adjusted_mutual_info_score: %f'%metrics.adjusted_mutual_info_score(y,  labels))
    print('silhouette_score: %f'%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"kmeans_PCA_metrics.txt"
    file = open(results, "w")
    file.write("K-Means clustering on the PCA-reduced data\n")
    file.write("Homogeneity Score: %f\n"%metrics.homogeneity_score(y, labels))
    file.write("Completeness Score: %f\n"%metrics.completeness_score(y, labels))
    file.write("V-Measure: %f\n"%metrics.v_measure_score(y, labels))
    file.write("The adjusted Rand index: %f\n"%metrics.adjusted_rand_score(y, labels))
    file.write("Adjusted Mutual Information: %f\n"%metrics.adjusted_mutual_info_score(y,  labels))
    file.write("Silhouette Score: %f\n"%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    file.write("\n")
    file.write("True Value, Clusters, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],labels[n],(n+1)))
    file.close()
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max]
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() , reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min() , reduced_data[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=pl.cm.Paired,
              aspect='auto', origin='lower')
    pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = k_means.cluster_centers_
    pl.scatter(centroids[:, 0], centroids[:, 1],
               marker='x', s=169, linewidths=3,
               color='w', zorder=10)
    pl.title('K-means clustering on the PCA-reduced data\n'
             'Number of clusters: %i'%n_clusters)
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
    save = Output + "kmeans_PCA.png"
    pl.savefig(save)
    lvltrace.lvltrace("LVLSortie dans KMeans_PCA unsupervised")

############################################# Mean-shift clustering ############################################
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

def meanshift(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans meanshift unsupervised")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    sample_size, n_features = X.shape
    # Compute clustering with MeanShift
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=sample_size)
    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print "#########################################################################################################\n"
    print "Mean Shift"
    print("number of estimated clusters : %d" % n_clusters_)
    #print labels
    #print y
    print('homogeneity_score: %f'%metrics.homogeneity_score(y, labels))
    print('completeness_score: %f'%metrics.completeness_score(y, labels))
    print('v_measure_score: %f'%metrics.v_measure_score(y, labels))
    print('adjusted_rand_score: %f'%metrics.adjusted_rand_score(y, labels))
    print('adjusted_mutual_info_score: %f'%metrics.adjusted_mutual_info_score(y,  labels))
    try:
        print('silhouette_score: %f'%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    except (ValueError):
        print "ValueError: Number of labels is 1 but should be more than 2and less than n_samples - 1"
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"mean_shift_metrics.txt"
    file = open(results, "w")
    file.write("Mean Shift\n")
    file.write("Homogeneity Score: %f\n"%metrics.homogeneity_score(y, labels))
    file.write("Completeness Score: %f\n"%metrics.completeness_score(y, labels))
    file.write("V-Measure: %f\n"%metrics.v_measure_score(y, labels))
    file.write("The adjusted Rand index: %f\n"%metrics.adjusted_rand_score(y, labels))
    file.write("Adjusted Mutual Information: %f\n"%metrics.adjusted_mutual_info_score(y,  labels))
    try:
        file.write("Silhouette Score: %f\n"%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    except (ValueError):
        file.write("ValueError: Number of labels is 1 but should be more than 2and less than n_samples - 1")
    file.write("\n")
    file.write("True Value, Clusters, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],labels[n],(n+1)))
    file.close()

    # Plot result
    import pylab as pl
    from itertools import cycle
    fig, ax = plt.subplots()
    im=ax.scatter(X[:, 0], X[:, 1], c=labels, marker='.')
    for k in xrange(n_clusters_):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        #print cluster_center[0], cluster_center[1]
        ax.plot(cluster_center[0], cluster_center[1], 'w', color='b',
                marker='x', markersize=6)
    fig.colorbar(im);
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    save = Output + "mean_shift.png"
    plt.savefig(save)
    lvltrace.lvltrace("LVLSortie dans meanshift unsupervised")

############################################# Affinity Propagation #############################################
from sklearn.metrics import euclidean_distances
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn import cluster, covariance, manifold
import scipy
from sklearn.cluster.affinity_propagation_ import AffinityPropagation, \
    affinity_propagation

def affinitypropagation(input_file,type,pref,Output):
    lvltrace.lvltrace("LVLEntree dans affinitypropagation unsupervised")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))

    X = data[:,1:]
    #print (" ici X vaut ")
    #print X
    #print (" fin de print X")
    labels_true = data[:,0]
    # A tester
    if type == 'spearmanr':
        X = scipy.stats.stats.spearmanr(X,axis=1)[0]
    else:
        if type == 'euclidean':
            X = -euclidean_distances(X, squared=True)
        else:
            print "something wrong"
    if pref == 'median':
        # A tester entre min ou median
        preference = np.median(X)
    else:
        if pref == 'mean':
            preference = np.mean(X)
        else:
            if pref == 'min':
                preference = np.min(X)
            else:
                print "something wrong"
    print "#########################################################################################################\n"
    print "Affinity Propagation"
    print preference
    n_samples, n_features = X.shape
    cluster_centers_indices, labels = affinity_propagation(X, preference=preference)
    #print cluster_centers_indices
    n_clusters_ = len(cluster_centers_indices)
    n_clusters_ = len(cluster_centers_indices)
    #print labels_true
    #print labels
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"affinity_propagation.txt"
    file = open(results, "w")
    file.write("Affinity Propagation\n")
    file.write("Homogeneity Score: %f\n"%metrics.homogeneity_score(labels_true, labels))
    file.write("Completeness Score: %f\n"%metrics.completeness_score(labels_true, labels))
    file.write("V-Measure: %f\n"%metrics.v_measure_score(labels_true, labels))
    file.write("The adjusted Rand index: %f\n"%metrics.adjusted_rand_score(labels_true, labels))
    file.write("Adjusted Mutual Information: %f\n"%metrics.adjusted_mutual_info_score(labels_true,  labels))
    file.write("Silhouette Score: %f\n"%metrics.silhouette_score(X, labels, metric='sqeuclidean'))
    file.write("\n")
    file.write("True Value, Clusters, Iteration\n")
    for n in xrange(len(labels_true)):
        file.write("%f,%f,%i\n"%(labels_true[n],labels[n],(n+1)))
    file.close()
    
    # Plot result
    import pylab as pl
    from itertools import cycle
    pl.close('all')
    pl.figure(1)
    pl.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbg')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        pl.plot(X[class_members, 0], X[class_members, 1], col + '.')
        pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            pl.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    pl.title('Estimated number of clusters: %d' % n_clusters_)
    save = Output + "affinity_propagation.png"
    plt.savefig(save)
    lvltrace.lvltrace("LVLSortie dans affinitypropagation unsupervised")

#################### Hierarchical clustering: DBSCAN ###################################


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def dbscan(input_file, Output):
    lvltrace.lvltrace("LVLEntree dans dbscan unsupervised")
    # Generate sample data
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    labels_true = data[:,0]
    #X = StandardScaler().fit_transform(Y)
    # Compute DBSCAN
    db = DBSCAN().fit(X)
    core_samples = db.core_sample_indices_
    labels = db.labels_
    print "#########################################################################################################\n"
    print "DBSCAN"
    print labels_true
    print labels
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    print "\n"
    print "#########################################################################################################\n"
    results = Output+"dbscan.txt"
    file = open(results, "w")
    file.write("DBSCAN\n")
    file.write("Homogeneity Score: %f\n"%metrics.homogeneity_score(y, labels))
    file.write("Completeness Score: %f\n"%metrics.completeness_score(y, labels))
    file.write("V-Measure: %f\n"%metrics.v_measure_score(y, labels))
    file.write("The adjusted Rand index: %f\n"%metrics.adjusted_rand_score(y, labels))
    file.write("Adjusted Mutual Information: %f\n"%metrics.adjusted_mutual_info_score(y,  labels))
    file.write("Silhouette Score: %f\n"%metrics.silhouette_score(X, labels, metric='euclidean', sample_size=sample_size))
    file.write("\n")
    file.write("True Value, Clusters, Iteration\n")
    for n in xrange(len(y)):
        file.write("%f,%f,%i\n"%(y[n],labels[n],(n+1)))
    file.close()
    
    # Plot result
    import pylab as pl
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 6
        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_core_samples = [index for index in core_samples
                                if labels[index] == k]
        for index in class_members:
            x = X[index]
            if index in core_samples and k != -1:
                markersize = 14
            else:
                markersize = 6
            pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=markersize)
    pl.title('Estimated number of clusters: %d' % n_clusters_)
    save = Output + "dbscan.png"
    plt.savefig(save)
    lvltrace.lvltrace("LVLSortie dans dbscan unsupervised")

############################### GMM classification ####################################

import pylab as pl
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
import itertools

import numpy as np
from scipy import linalg
import pylab as pl
import matplotlib as mpl
from sklearn import mixture


def gmm(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans gmm unsupervised")
    print "#########################################################################################################\n"
    print "GMM"
    print "#########################################################################################################\n"
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    
    # Fit a mixture of gaussians with EM using five components
    gmm = mixture.GMM(n_components=5, covariance_type='spherical', init_params = 'wmc')
    gmm.fit(X)

    # Fit a dirichlet process mixture of gaussians using five components
    dpgmm = mixture.DPGMM(n_components=5, covariance_type='spherical',init_params = 'wmc')
    dpgmm.fit(X)

    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k'])

    for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                      (dpgmm, 'Dirichlet Process GMM')]):
        splot = pl.subplot(2, 1, 1 + i)
        Y_ = clf.predict(X)
        for i, (mean, covar, color) in enumerate(zip(
                                                     clf.means_, clf._get_covars(), color_iter)):
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            pl.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
            
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        pl.xticks(())
        pl.yticks(())
        pl.title(title)
    save = Output + "gmm.png"
    plt.savefig(save)
    lvltrace.lvltrace("LVLSortie dans gmm unsupervised")



############################################## DIMENSION REDUCTION ###########################################
#
#
####################################################### PCA ##################################################

def pca(input_file,Output):
    lvltrace.lvltrace("LVLEntree dans pca unsupervised")
    ncol=tools.file_col_coma(input_file)
    data = np.loadtxt(input_file, delimiter=',', usecols=range(ncol-1))
    X = data[:,1:]
    y = data[:,0]
    n_samples, n_features = X.shape
    # instantiate the model
    model = PCA(n_components=2)
    # fit the model: notice we don't pass the labels!
    model.fit(X)
    # transform the data to two dimensions
    X_PCA = model.transform(X)
    print "#########################################################################################################\n"
    print "PCA"
    print "shape of result:", X_PCA.shape
    print model.explained_variance_ratio_
    print "#########################################################################################################\n"
    
    results = Output+"pca.txt"
    file = open(results, "w")
    file.write("PCA\n")
    file.write("shape of result: %f,%f\n"%(X_PCA.shape[0],X_PCA.shape[1]))
    file.write("Explained variance ratio: %f,%f\n"%(model.explained_variance_ratio_[0],model.explained_variance_ratio_[1]))
    file.close()
    
    # plot the results along with the labels
    fig, ax = plt.subplots()
    im = ax.scatter(X_PCA[:, 0], X_PCA[:, 1], c=y)
    fig.colorbar(im);
    save = Output + "pca.png"
    plt.savefig(save)
    lvltrace.lvltrace("LVLSortie dans pca unsupervised")


