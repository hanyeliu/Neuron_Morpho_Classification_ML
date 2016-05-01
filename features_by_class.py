# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

import os,random
import lvltrace
import sys
import shutil
import input_measure as im
import inputs
import csv
import tools


##############################################################
# Put extracted features to the output folder
# output are csv files with coma separated variables
# This function skip empty files
##############################################################

def features_by_class():
    lvltrace.lvltrace("LVLEntree dans features_by_class")
    if not os.path.exists(inputs.features_output):
        os.makedirs(inputs.features_output)
    for root, dirs, files in os.walk(inputs.morphology):
        for i in dirs:
            #LVlprint ("on traite le repertoire "+ str(i))
            neuron_dir=root+'/'+i
            neuron_file_out=inputs.features_output+'/'+i+'.csv'
            features_by_class = open(neuron_file_out, "w")
            writer=csv.writer(features_by_class, lineterminator='\t')
            features_name=tools.random_file(neuron_dir)
            lines=tools.file_lines(features_name)
            features_by_class.write("mtype\tneuron_name\t")
            for line in xrange(lines):
                features_by_class.write("%s\t"%tools.read_csv_tab(features_name,1,line))
            features_by_class.write("\n")

            for file in os.listdir(neuron_dir):
                neuron_file_in=root+'/'+i+'/'+file
                # if the extracted feature file from lmeasure is empty, then skip
                if file.endswith(".csv") and os.path.getsize(neuron_file_in) > 0:
                    lines=tools.file_lines(neuron_file_in)
                    features_by_class.write("%s\t"%i)
                    features_by_class.write("%s\t"%file)
                    for line in xrange(lines):
                        features_by_class.write("%s\t"%tools.read_csv_tab(neuron_file_in,2,line))
                    features_by_class.write("\n")
    lvltrace.lvltrace("LVLSortie dans features_by_class")
