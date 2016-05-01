# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

#  input_files.py
#  To automatically create the output folders
#  
#
#  Updated by Xavier Vasques on 19/02/2015.
#

import os
import inputs
import configuration

###################################### MAPS ############################

Output_folder_descriptive=inputs.data_analysis+'Descriptive_analysis/'
Output_folder_descriptive_preproc=Output_folder_descriptive+'Data_Preprocessing/'
Coma_Features_descriptive=Output_folder_descriptive_preproc+'Features_coma/'
Corrected_Features_descriptive=Output_folder_descriptive_preproc+'Corrected_Features/'
Results_descriptive=Output_folder_descriptive+'Results/'
descriptive_analysis_output=Results_descriptive+'Descriptive/'
descriptive_variability=Results_descriptive+'Variability/'

if not os.path.exists(Output_folder_descriptive):
    os.makedirs(Output_folder_descriptive)
if not os.path.exists(Output_folder_descriptive_preproc):
    os.makedirs(Output_folder_descriptive_preproc)
if not os.path.exists(Coma_Features_descriptive):
    os.makedirs(Coma_Features_descriptive)
if not os.path.exists(Corrected_Features_descriptive):
    os.makedirs(Corrected_Features_descriptive)
if not os.path.exists(Results_descriptive):
    os.makedirs(Results_descriptive)
if not os.path.exists(descriptive_analysis_output):
    os.makedirs(descriptive_analysis_output)
if not os.path.exists(descriptive_variability):
    os.makedirs(descriptive_variability)

for the_file in os.listdir(descriptive_variability):
    file_path = os.path.join(descriptive_variability, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(descriptive_analysis_output):
    file_path = os.path.join(descriptive_analysis_output, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(Results_descriptive):
    file_path = os.path.join(Results_descriptive, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(Output_folder_descriptive):
    file_path = os.path.join(Output_folder_descriptive, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e
for the_file in os.listdir(Output_folder_descriptive_preproc):
    file_path = os.path.join(Output_folder_descriptive_preproc, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e
for the_file in os.listdir(Coma_Features_descriptive):
    file_path = os.path.join(Coma_Features_descriptive, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e
for the_file in os.listdir(Corrected_Features_descriptive):
    file_path = os.path.join(Corrected_Features_descriptive, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e

#########################################################################



input_data=inputs.data_analysis+'Classification/Data_Preprocessing/Merged_files/features.csv'
input_data_features_selection=inputs.data_analysis+'Features_selection/Data_Preprocessing/Merged_files/features.csv'
input_data_features_selection_glm = inputs.data_analysis+'Features_selection/Data_Preprocessing_glm/Merged_files/features.csv'
output_supervised = inputs.data_analysis+'Classification/Supervised/'
output_unsupervised = inputs.data_analysis+'Classification/Unsupervised/'
output_features_selection = inputs.data_analysis+'Features_selection/Results/'

# The folder where the extracted features are
Extracted_Features=inputs.features_output

# The folder you want to create for results
Output_folder = inputs.data_analysis+'Classification/Data_Preprocessing/'
Output_folder_Supervised=inputs.data_analysis+'Classification/Supervised/'
Output_folder_Unsupervised=inputs.data_analysis+'Classification/Unsupervised/'
Output_folder_Features_Selection=inputs.data_analysis+'Features_selection/Data_Preprocessing/'
Output_folder_Features_Selection_glm=inputs.data_analysis+'Features_selection/Data_Preprocessing_glm/'

norm=configuration.normalization #(binarize,normalize,standardize, scaling)

# Select the name of a file in the Extracted Features Folder
file_name=configuration.file_name


#################################################################################################

#               Outputs folders creation/delete contents in Outputs folders                     #

#################################################################################################

# Create output folders for Features Selection: GLM

Coma_Features_fs_glm=Output_folder_Features_Selection_glm+'Features_coma/'
Corrected_Features_fs_glm=Output_folder_Features_Selection_glm+'Features_corr/'
Merged_files_fs_glm=Output_folder_Features_Selection_glm+'Merged_files/'
Preprocessed_file_fs_glm=Output_folder_Features_Selection_glm+'Merged_files/features.csv'
similarity_fs_glm=Output_folder_Features_Selection_glm+'Merged_files/similarity.csv'
ontology_fs_glm=Output_folder_Features_Selection_glm+'Merged_files/ontology.csv'

if not os.path.exists(Output_folder_Features_Selection_glm):
    os.makedirs(Output_folder_Features_Selection_glm)
if not os.path.exists(Coma_Features_fs_glm):
    os.makedirs(Coma_Features_fs_glm)
if not os.path.exists(Corrected_Features_fs_glm):
    os.makedirs(Corrected_Features_fs_glm)
if not os.path.exists(Merged_files_fs_glm):
    os.makedirs(Merged_files_fs_glm)
if not os.path.exists(Coma_Features_fs_glm):
    os.makedirs(Coma_Features_fs_glm)
if not os.path.exists(Corrected_Features_fs_glm):
    os.makedirs(Corrected_Features_fs_glm)
if not os.path.exists(Merged_files_fs_glm):
    os.makedirs(Merged_files_fs_glm)




# Create output folders for Features Selection: forest of tree and univariate

Coma_Features_fs=Output_folder_Features_Selection+'Features_coma/'
Corrected_Features_fs=Output_folder_Features_Selection+'Features_corr/'
Merged_files_fs=Output_folder_Features_Selection+'Merged_files/'
Preprocessed_file_fs=Output_folder_Features_Selection+'Merged_files/features.csv'
similarity_fs=Output_folder_Features_Selection+'Merged_files/similarity.csv'
ontology_fs=Output_folder_Features_Selection+'Merged_files/ontology.csv'
Results_fs=output_features_selection

if not os.path.exists(Output_folder_Features_Selection):
    os.makedirs(Output_folder_Features_Selection)
if not os.path.exists(Coma_Features_fs):
    os.makedirs(Coma_Features_fs)
if not os.path.exists(Corrected_Features_fs):
    os.makedirs(Corrected_Features_fs)
if not os.path.exists(Merged_files_fs):
    os.makedirs(Merged_files_fs)
if not os.path.exists(Coma_Features_fs):
    os.makedirs(Coma_Features_fs)
if not os.path.exists(Corrected_Features_fs):
    os.makedirs(Corrected_Features_fs)
if not os.path.exists(Merged_files_fs):
    os.makedirs(Merged_files_fs)
if not os.path.exists(Results_fs):
    os.makedirs(Results_fs)



# Create output folders for supervised and unsupervised algorithms

Coma_Features=Output_folder+'Features_coma/'
Corrected_Features=Output_folder+'Features_corr/'
Merged_files=Output_folder+'Merged_files/'
Preprocessed_file=Output_folder+'Merged_files/features.csv'
similarity=Output_folder+'Merged_files/similarity.csv'
ontology=Output_folder+'Merged_files/ontology.csv'
Results_sup=Output_folder_Supervised+'/'
Results_unsup=Output_folder_Unsupervised+'/'

if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)
if not os.path.exists(Coma_Features):
    os.makedirs(Coma_Features)
if not os.path.exists(Corrected_Features):
    os.makedirs(Corrected_Features)
if not os.path.exists(Merged_files):
    os.makedirs(Merged_files)
if not os.path.exists(Coma_Features):
    os.makedirs(Coma_Features)
if not os.path.exists(Corrected_Features):
    os.makedirs(Corrected_Features)
if not os.path.exists(Merged_files):
    os.makedirs(Merged_files)
if not os.path.exists(Results_sup):
    os.makedirs(Results_sup)
if not os.path.exists(Results_unsup):
    os.makedirs(Results_unsup)

#### GLM
#
#
## Delete the contents of the output folders
#for the_file in os.listdir(Output_folder_Features_Selection_glm):
#    file_path = os.path.join(Output_folder_Features_Selection_glm, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Coma_Features_fs_glm):
#    file_path = os.path.join(Coma_Features_fs_glm, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Corrected_Features_fs_glm):
#    file_path = os.path.join(Corrected_Features_fs_glm, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Merged_files_fs_glm):
#    file_path = os.path.join(Merged_files_fs_glm, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#### Data Preprocessing classification
#
## Delete the contents of the output folders
#for the_file in os.listdir(Output_folder):
#    file_path = os.path.join(Output_folder, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Coma_Features):
#    file_path = os.path.join(Coma_Features, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Corrected_Features):
#    file_path = os.path.join(Corrected_Features, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Merged_files):
#    file_path = os.path.join(Merged_files, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#
#
##### Features selection
#
#
## Delete the contents of the output folders
#for the_file in os.listdir(Output_folder_Features_Selection):
#    file_path = os.path.join(Output_folder_Features_Selection, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Coma_Features_fs):
#    file_path = os.path.join(Coma_Features_fs, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Corrected_Features_fs):
#    file_path = os.path.join(Corrected_Features_fs, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Merged_files_fs):
#    file_path = os.path.join(Merged_files_fs, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#for the_file in os.listdir(Results_fs):
#    file_path = os.path.join(Results_fs, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#
#### supervised
#
#for the_file in os.listdir(Results_sup):
#    file_path = os.path.join(Results_sup, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#### unsupervised
#
#for the_file in os.listdir(Results_unsup):
#    file_path = os.path.join(Results_unsup, the_file)
#    try:
#        if os.path.isfile(file_path):
#            os.unlink(file_path)
#    except Exception, e:
#        print e
#
#
#
#
#
