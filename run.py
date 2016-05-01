# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

#  run.py
#  Main script to run (python run.py)
#  This script run all the needed function requested by configuration.py
#
#  Updated by Xavier Vasques on 19/02/2015.
#

import os
import lvltrace
import inputs
import input_files
import configuration
import features_extraction
import features_by_class
import supervised
import unsupervised
import data_preprocessing
import supervised_split_test
import data_preproc_features_select
import features_selection
import data_preprocessing_descriptive
import descriptive_analysis
import features_variability


if configuration.only_extract == 'ON':
    if configuration.extract_analyse == 'ON':
        print "In configuration.py file, 'only_extract' and 'extract_analyse' can not be both ON"
    else:
        features_extraction.features_extraction()
        features_by_class.features_by_class()
else:
    if configuration.extract_analyse == 'OFF':
        print "Nothing To Do"
    else:
        if configuration.already_extracted == 'NO':
            features_extraction.features_extraction()
            features_by_class.features_by_class()
            
            if configuration.gaussian_nb == 'ON' or configuration.multinomial_nb == 'ON' or configuration.kneighbors == 'ON' or configuration.radius_neighbors == 'ON' or configuration.lda == 'ON' or configuration.qda == 'ON' or configuration.svc_rbf == 'ON' or configuration.svc_linear == 'ON' or configuration.stochastic_gd == 'ON' or configuration.nearest_centroid == 'ON' or configuration.decisiontree == 'ON' or configuration.randomforest == 'ON' or configuration.extratreeclassifier == 'ON' or configuration.gaussian_nb_test == 'ON' or configuration.multinomial_nb_test == 'ON' or configuration.kneighbors_test == 'ON' or configuration.radius_neighbors_test == 'ON' or configuration.lda_test == 'ON' or configuration.qda_test == 'ON' or configuration.svc_rbf_test == 'ON' or configuration.svc_linear_test == 'ON' or configuration.stochastic_gd_test == 'ON' or configuration.nearest_centroid_test == 'ON' or configuration.decisiontree_test == 'ON' or configuration.randomforest_test == 'ON' or configuration.extratreeclassifier_test == 'ON' or configuration.kmeans == 'ON' or configuration.MiniBatchKMeans == 'ON' or configuration.KMeans_PCA == 'ON' or configuration.ward == 'ON' or configuration.meanshift == 'ON' or configuration.affinitypropagation == 'ON' or configuration.gmm == 'ON' or configuration.pca == 'ON':
            
                data_preprocessing.preprocessing_module(input_files.Extracted_Features,input_files.Coma_Features,input_files.Corrected_Features,input_files.norm,input_files.ontology)
                data_preprocessing.merger(input_files.Preprocessed_file,input_files.file_name,input_files.Corrected_Features)
                data_preprocessing.neuron_similarity_matrix_labelled(input_files.Preprocessed_file,input_files.similarity,input_files.Corrected_Features)
                
                # Supervised with training and test data set
                if configuration.gaussian_nb == 'ON':
                    supervised.gaussianNB(input_files.input_data,input_files.output_supervised)
                if configuration.multinomial_nb == 'ON':
                    supervised.multinomialNB(input_files.input_data,input_files.output_supervised)
                if configuration.kneighbors == 'ON':
                    supervised.kneighbors(input_files.input_data,input_files.output_supervised)
                if configuration.radius_neighbors == 'ON':
                    supervised.Radius_Neighbors(input_files.input_data,input_files.output_supervised)
                if configuration.lda == 'ON':
                    supervised.lda(input_files.input_data,input_files.output_supervised)
                if configuration.qda == 'ON':
                    supervised.qda(input_files.input_data,input_files.output_supervised)
                if configuration.svc_rbf == 'ON':
                    supervised.SVC_rbf(input_files.input_data,input_files.output_supervised)
                if configuration.svc_linear == 'ON':
                    supervised.SVC_linear(input_files.input_data,input_files.output_supervised)
                if configuration.stochastic_gd == 'ON':
                    supervised.stochasticGD(input_files.input_data,input_files.output_supervised)
                if configuration.nearest_centroid == 'ON':
                    supervised.nearest_centroid(input_files.input_data,input_files.output_supervised)
                if configuration.decisiontree == 'ON':
                    supervised.decisiontreeclassifier(input_files.input_data,input_files.output_supervised)
                if configuration.randomforest == 'ON':
                    supervised.randomforest(input_files.input_data,input_files.output_supervised)
                if configuration.extratreeclassifier == 'ON':
                    supervised.extratreeclassifier(input_files.input_data,input_files.output_supervised)
                
                # Supervised with training and test data set
            
                if configuration.gaussian_nb_test == 'ON':
                    supervised_split_test.gaussianNB(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.multinomial_nb_test == 'ON':
                    supervised_split_test.multinomialNB(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.kneighbors_test == 'ON':
                    supervised_split_test.kneighbors(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.radius_neighbors_test == 'ON':
                    supervised_split_test.Radius_Neighbors(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.lda_test == 'ON':
                    supervised_split_test.lda(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.qda_test == 'ON':
                    supervised_split_test.qda(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.svc_rbf_test == 'ON':
                    supervised_split_test.SVC_rbf(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.svc_linear_test == 'ON':
                    supervised_split_test.SVC_linear(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.stochastic_gd_test == 'ON':
                    supervised_split_test.stochasticGD(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.nearest_centroid_test == 'ON':
                    supervised_split_test.nearest_centroid(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.decisiontree_test == 'ON':
                    supervised_split_test.decisiontreeclassifier(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.randomforest_test == 'ON':
                    supervised_split_test.randomforest(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.extratreeclassifier_test == 'ON':
                    supervised_split_test.extratreeclassifier(input_files.input_data,input_files.output_supervised,configuration.testing_set)

                # Unsupervised
                if configuration.kmeans == 'ON':
                    unsupervised.kmeans(input_files.input_data,configuration.n_clusters,input_files.output_unsupervised)
                if configuration.MiniBatchKMeans == 'ON':
                    unsupervised.MiniBatchKMeans(input_files.input_data,configuration.n_clusters,input_files.output_unsupervised)
                if configuration.KMeans_PCA == 'ON':
                    unsupervised.KMeans_PCA(input_files.input_data,configuration.n_clusters,input_files.output_unsupervised)
                #if configuration.ward == 'ON':
                    #unsupervised.ward(input_files.input_data,configuration.n_clusters,configuration.n_neighbors,input_files.output_unsupervised)
                if configuration.meanshift == 'ON':
                    unsupervised.meanshift(input_files.input_data,input_files.output_unsupervised)
                if configuration.affinitypropagation == 'ON':
                    unsupervised.affinitypropagation(input_files.input_data,configuration.type,configuration.pref,input_files.output_unsupervised)
                #if configuration.dbscan == 'ON':
                    #unsupervised.dbscan(input_files.input_data,input_files.output_unsupervised)
                if configuration.gmm == 'ON':
                    unsupervised.gmm(input_files.input_data,input_files.output_unsupervised)
                if configuration.pca == 'ON':
                    unsupervised.pca(input_files.input_data,input_files.output_unsupervised)

            # Features selection
            if configuration.univariate == 'ON' or configuration.forest_of_trees == 'ON':
                data_preproc_features_select.preprocessing_module(input_files.Extracted_Features,input_files.Coma_Features_fs,input_files.Corrected_Features_fs,input_files.norm,input_files.ontology_fs)
                data_preproc_features_select.merger(input_files.Preprocessed_file_fs,input_files.file_name,input_files.Corrected_Features_fs) #with features labels
                data_preproc_features_select.neuron_similarity_matrix_labelled(input_files.Preprocessed_file_fs,input_files.similarity_fs,input_files.Corrected_Features_fs) # with class labels
                if configuration.univariate == 'ON':
                        features_selection.univariate(input_files.input_data_features_selection, input_files.output_features_selection,configuration.percentile)
                if configuration.forest_of_trees == 'ON':
                        features_selection.forest_of_trees(input_files.input_data_features_selection,input_files.output_features_selection)
            configuration.predict_class_glm = 'OFF'
            if configuration.predict_class_glm == 'ON':
                data_preproc_features_select.preprocessing_module(input_files.Extracted_Features,input_files.Coma_Features_fs_glm,input_files.Corrected_Features_fs_glm,input_files.norm,input_files.ontology_fs_glm)
                data_preproc_features_select.merger_labelled(input_files.Preprocessed_file_fs_glm,input_files.file_name,input_files.Corrected_Features_fs_glm) #with features labels
                data_preproc_features_select.neuron_similarity_matrix_labelled(input_files.Preprocessed_file_fs_glm,input_files.similarity_fs_glm,input_files.Corrected_Features_fs_glm) # with class labels
                features_selection.predict_class_glm(input_files.input_data_features_selection_glm,input_files.output_features_selection)

            # Descriptive Analysis
            if configuration.descriptive == 'ON':
                data_preprocessing_descriptive.data_preprocessing_descriptive(input_files.Extracted_Features,input_files.Coma_Features_descriptive,input_files.Corrected_Features_descriptive)
                descriptive_analysis.descriptive_analysis(input_files.Extracted_Features,input_files.descriptive_analysis_output,configuration.cores,input_files.Corrected_Features_descriptive)
                features_variability.features_variability(input_files.descriptive_analysis_output,input_files.descriptive_variability,configuration.cores)

        else:
            if configuration.gaussian_nb == 'ON' or configuration.multinomial_nb == 'ON' or configuration.kneighbors == 'ON' or configuration.radius_neighbors == 'ON' or configuration.lda == 'ON' or configuration.qda == 'ON' or configuration.svc_rbf == 'ON' or configuration.svc_linear == 'ON' or configuration.stochastic_gd == 'ON' or configuration.nearest_centroid == 'ON' or configuration.decisiontree == 'ON' or configuration.randomforest == 'ON' or configuration.extratreeclassifier == 'ON' or configuration.gaussian_nb_test == 'ON' or configuration.multinomial_nb_test == 'ON' or configuration.kneighbors_test == 'ON' or configuration.radius_neighbors_test == 'ON' or configuration.lda_test == 'ON' or configuration.qda_test == 'ON' or configuration.svc_rbf_test == 'ON' or configuration.svc_linear_test == 'ON' or configuration.stochastic_gd_test == 'ON' or configuration.nearest_centroid_test == 'ON' or configuration.decisiontree_test == 'ON' or configuration.randomforest_test == 'ON' or configuration.extratreeclassifier_test == 'ON' or configuration.kmeans == 'ON' or configuration.MiniBatchKMeans == 'ON' or configuration.KMeans_PCA == 'ON' or configuration.ward == 'ON' or configuration.meanshift == 'ON' or configuration.affinitypropagation == 'ON' or configuration.gmm == 'ON' or configuration.pca == 'ON':
            
                data_preprocessing.preprocessing_module(input_files.Extracted_Features,input_files.Coma_Features,input_files.Corrected_Features,input_files.norm,input_files.ontology)
                data_preprocessing.merger(input_files.Preprocessed_file,input_files.file_name,input_files.Corrected_Features)
                data_preprocessing.neuron_similarity_matrix_labelled(input_files.Preprocessed_file,input_files.similarity,input_files.Corrected_Features)
                
                # Supervised without training and test data set
                
                if configuration.gaussian_nb == 'ON':
                    supervised.gaussianNB(input_files.input_data,input_files.output_supervised)
                if configuration.multinomial_nb == 'ON':
                    supervised.multinomialNB(input_files.input_data,input_files.output_supervised)
                if configuration.kneighbors == 'ON':
                    supervised.kneighbors(input_files.input_data,input_files.output_supervised)
                if configuration.radius_neighbors == 'ON':
                    supervised.Radius_Neighbors(input_files.input_data,input_files.output_supervised)
                if configuration.lda == 'ON':
                    supervised.lda(input_files.input_data,input_files.output_supervised)
                if configuration.qda == 'ON':
                    supervised.qda(input_files.input_data,input_files.output_supervised)
                if configuration.svc_rbf == 'ON':
                    supervised.SVC_rbf(input_files.input_data,input_files.output_supervised)
                if configuration.svc_linear == 'ON':
                    supervised.SVC_linear(input_files.input_data,input_files.output_supervised)
                if configuration.stochastic_gd == 'ON':
                    supervised.stochasticGD(input_files.input_data,input_files.output_supervised)
                if configuration.nearest_centroid == 'ON':
                    supervised.nearest_centroid(input_files.input_data,input_files.output_supervised)
                if configuration.decisiontree == 'ON':
                    supervised.decisiontreeclassifier(input_files.input_data,input_files.output_supervised)
                if configuration.randomforest == 'ON':
                    supervised.randomforest(input_files.input_data,input_files.output_supervised)
                if configuration.extratreeclassifier == 'ON':
                    supervised.extratreeclassifier(input_files.input_data,input_files.output_supervised)

                # Supervised with training and test data set

                if configuration.gaussian_nb_test == 'ON':
                    supervised_split_test.gaussianNB(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.multinomial_nb_test == 'ON':
                    supervised_split_test.multinomialNB(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.kneighbors_test == 'ON':
                    supervised_split_test.kneighbors(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.radius_neighbors_test == 'ON':
                    supervised_split_test.Radius_Neighbors(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.lda_test == 'ON':
                    supervised_split_test.lda(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.qda_test == 'ON':
                    supervised_split_test.qda(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.svc_rbf_test == 'ON':
                    supervised_split_test.SVC_rbf(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.svc_linear_test == 'ON':
                    supervised_split_test.SVC_linear(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.stochastic_gd_test == 'ON':
                    supervised_split_test.stochasticGD(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.nearest_centroid_test == 'ON':
                    supervised_split_test.nearest_centroid(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.decisiontree_test == 'ON':
                    supervised_split_test.decisiontreeclassifier(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.randomforest_test == 'ON':
                    supervised_split_test.randomforest(input_files.input_data,input_files.output_supervised,configuration.testing_set)
                if configuration.extratreeclassifier_test == 'ON':
                    supervised_split_test.extratreeclassifier(input_files.input_data,input_files.output_supervised,configuration.testing_set)

                # Unsupervised
                if configuration.kmeans == 'ON':
                    unsupervised.kmeans(input_files.input_data,configuration.n_clusters,input_files.output_unsupervised)
                if configuration.MiniBatchKMeans == 'ON':
                    unsupervised.MiniBatchKMeans(input_files.input_data,configuration.n_clusters,input_files.output_unsupervised)
                if configuration.KMeans_PCA == 'ON':
                    unsupervised.KMeans_PCA(input_files.input_data,configuration.n_clusters,input_files.output_unsupervised)
                #if configuration.ward == 'ON':
                    #unsupervised.ward(input_files.input_data,configuration.n_clusters,configuration.n_neighbors,input_files.output_unsupervised)
                if configuration.meanshift == 'ON':
                    unsupervised.meanshift(input_files.input_data,input_files.output_unsupervised)
                if configuration.affinitypropagation == 'ON':
                    unsupervised.affinitypropagation(input_files.input_data,configuration.type,configuration.pref,input_files.output_unsupervised)
                #if configuration.dbscan == 'ON':
                    #unsupervised.dbscan(input_files.input_data,input_files.output_unsupervised)
                if configuration.gmm == 'ON':
                    unsupervised.gmm(input_files.input_data,input_files.output_unsupervised)
                if configuration.pca == 'ON':
                    unsupervised.pca(input_files.input_data,input_files.output_unsupervised)

            # Features selection
            if configuration.univariate == 'ON' or configuration.forest_of_trees == 'ON':
                data_preproc_features_select.preprocessing_module(input_files.Extracted_Features,input_files.Coma_Features_fs,input_files.Corrected_Features_fs,input_files.norm,input_files.ontology_fs)
                data_preproc_features_select.merger(input_files.Preprocessed_file_fs,input_files.file_name,input_files.Corrected_Features_fs) #with features labels
                data_preproc_features_select.neuron_similarity_matrix_labelled(input_files.Preprocessed_file_fs,input_files.similarity_fs,input_files.Corrected_Features_fs) # with class labels
                if configuration.univariate == 'ON':
                    features_selection.univariate(input_files.input_data_features_selection, input_files.output_features_selection,configuration.percentile)
                if configuration.forest_of_trees == 'ON':
                        features_selection.forest_of_trees(input_files.input_data_features_selection,input_files.output_features_selection)
            configuration.predict_class_glm = 'OFF'
            if configuration.predict_class_glm == 'ON':
                data_preproc_features_select.preprocessing_module(input_files.Extracted_Features,input_files.Coma_Features_fs_glm,input_files.Corrected_Features_fs_glm,input_files.norm,input_files.ontology_fs_glm)
                data_preproc_features_select.merger_labelled(input_files.Preprocessed_file_fs_glm,input_files.file_name,input_files.Corrected_Features_fs_glm) #with features labels
                data_preproc_features_select.neuron_similarity_matrix_labelled(input_files.Preprocessed_file_fs_glm,input_files.similarity_fs_glm,input_files.Corrected_Features_fs_glm) # with class labels
                features_selection.predict_class_glm(input_files.input_data_features_selection_glm,input_files.output_features_selection)

            # Descriptive Analysis
            if configuration.descriptive == 'ON':
                data_preprocessing_descriptive.data_preprocessing_descriptive(input_files.Extracted_Features,input_files.Coma_Features_descriptive,input_files.Corrected_Features_descriptive)
                descriptive_analysis.descriptive_analysis(input_files.Extracted_Features,input_files.descriptive_analysis_output,configuration.cores,input_files.Corrected_Features_descriptive)
                features_variability.features_variability(input_files.descriptive_analysis_output,input_files.descriptive_variability,configuration.cores)
                

print "Sortie normale de run.py"






