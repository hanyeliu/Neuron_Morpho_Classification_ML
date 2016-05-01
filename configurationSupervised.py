# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

import os

only_extract='OFF' # Only extract feautures using lmeasure
extract_analyse='ON' # Analyze the morphologies
already_extracted='NO' # 'YES' or 'NO'
result_number ='1' # Result number in order to name the output folders differently (Data_Analysis_1, Data_Analysis_2 etc ...)
cores=4
count=0
debut=0

#######################################################################
#                   C L A S S I F I C A T I O N
#######################################################################

#################### DATA PREPROCESSING ###############################
# Data Normalization: Choose the type of normalization you want
# binarize = binarize the data
# normalize = normalize the data with axis = 0 and l2 norm
# standardize = standardize the data to [0,1]
# scaling the data

normalization='standardize' #(binarize,normalize,standardize, scaling)

# Select randomly the name of a file in the Extracted Features Folder
file_name='L4_PC.csv'


#################### SUPERVISED ALGORITHMS ############################

# Supervised without training set and testing set but only training set
gaussian_nb='OFF'
multinomial_nb='OFF'
kneighbors='OFF'
radius_neighbors='OFF'
lda='OFF'
qda='OFF'
svc_rbf='OFF'
svc_linear='OFF'
stochastic_gd='OFF'
nearest_centroid='OFF'
decisiontree='OFF'
randomforest='OFF'
extratreeclassifier='OFF'

# Supervised with training set and testing set
testing_set=0.3 # 0.1 means 10% of the training set is used as a testing set

gaussian_nb_test='ON'
multinomial_nb_test='ON'
kneighbors_test='ON'
radius_neighbors_test='ON'
lda_test='ON'
qda_test='ON'
svc_rbf_test='ON'
svc_linear_test='ON'
stochastic_gd_test='ON'
nearest_centroid_test='ON'
decisiontree_test='ON'
randomforest_test='ON'
extratreeclassifier_test='ON'

#################### UNSUPERVISED ALGORITHMS ############################

# Define the number of clusters for K-Means, MiniBatchKMeans, KMeans_PCA and Ward
n_clusters = 23

# For Ward: Define the structure A of the data. Here a 100 nearest neighbors
n_neighbors = 100

# For Affinitity Propagation
# type = 'spearmanr' or 'euclidean'
# preference = 'median' or 'min' or 'mean'
type = 'spearmanr'
pref = 'median'

kmeans = 'OFF'
MiniBatchKMeans = 'OFF'
KMeans_PCA = 'OFF'
ward = 'OFF'
meanshift = 'OFF'
affinitypropagation = 'OFF'
dbscan = 'OFF'
gmm = 'OFF'
pca = 'OFF'


#################### FEATURES SELECTION ############################

# Univariate feature selection with F-test for feature scoring
univariate = 'OFF'
# We use the default selection function: the 10% most significant features
percentile=13

forest_of_trees = 'OFF'
predict_class_glm = 'OFF'

#######################################################################
#                   D E S C R I P T I V E
#######################################################################

descriptive = 'OFF'

