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


# we start by extracting and classing morphologies
features_extraction.features_extraction()
features_by_class.features_by_class()
            
            
#Then we preprocess data
data_preprocessing.preprocessing_module(input_files.Extracted_Features,input_files.Coma_Features,input_files.Corrected_Features,input_files.norm,input_files.ontology)
data_preprocessing.merger(input_files.Preprocessed_file,input_files.file_name,input_files.Corrected_Features)
data_preprocessing.neuron_similarity_matrix_labelled(input_files.Preprocessed_file,input_files.similarity,input_files.Corrected_Features)
                
#The only supervised algorithm we focus on is lda
supervised.lda(input_files.input_data,input_files.output_supervised)

#The only unsupervised algorithm we focus on is affinity propagation
unsupervised.affinitypropagation(input_files.input_data,configuration.type,configuration.pref,input_files.output_unsupervised)

