#  delete_files.py
#  To automatically delete the output files
#
#
#  Updated by Xavier Vasques on 19/02/2015.
#

import os
import inputs
import configuration
import input_files

#### GLM

def delete_files_glm():
    # Delete the contents of the output folders
    for the_file in os.listdir(input_files.Output_folder_Features_Selection_glm):
        file_path = os.path.join(input_files.Output_folder_Features_Selection_glm, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

for the_file in os.listdir(input_files.Coma_Features_fs_glm):
    file_path = os.path.join(input_files.Coma_Features_fs_glm, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(input_files.Corrected_Features_fs_glm):
    file_path = os.path.join(input_files.Corrected_Features_fs_glm, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(input_files.Merged_files_fs_glm):
    file_path = os.path.join(input_files.Merged_files_fs_glm, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception, e:
        print e

### Data Preprocessing classification
def delete_files_data_preprocessing():
    # Delete the contents of the output folders
    for the_file in os.listdir(input_files.Output_folder):
        file_path = os.path.join(input_files.Output_folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception, e:
        print e

    for the_file in os.listdir(input_files.Coma_Features):
        file_path = os.path.join(input_files.Coma_Features, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        except Exception, e:
            print e

    for the_file in os.listdir(input_files.Corrected_Features):
        file_path = os.path.join(input_files.Corrected_Features, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        except Exception, e:
            print e

    for the_file in os.listdir(input_files.Merged_files):
        file_path = os.path.join(input_files.Merged_files, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        except Exception, e:
            print e



#### Features selection

def delete_files_feature_selection():
    # Delete the contents of the output folders
    for the_file in os.listdir(input_files.Output_folder_Features_Selection):
        file_path = os.path.join(input_files.Output_folder_Features_Selection, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

for the_file in os.listdir(input_files.Coma_Features_fs):
    file_path = os.path.join(input_files.Coma_Features_fs, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(input_files.Corrected_Features_fs):
    file_path = os.path.join(input_files.Corrected_Features_fs, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(input_files.Merged_files_fs):
    file_path = os.path.join(input_files.Merged_files_fs, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception, e:
        print e

for the_file in os.listdir(input_files.Results_fs):
    file_path = os.path.join(input_files.Results_fs, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception, e:
        print e


### supervised
def delete_files_sup():
    for the_file in os.listdir(input_files.Results_sup):
        file_path = os.path.join(input_files.Results_sup, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

### unsupervised
def delete_files_unsup():
    for the_file in os.listdir(input_files.Results_unsup):
        file_path = os.path.join(input_files.Results_unsup, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e