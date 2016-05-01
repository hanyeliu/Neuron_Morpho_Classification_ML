# Xavier Vasques
# Laboratoire de Recherche en Neurosciences Cliniques (LRENC)
# Update May, 1st 2016

#  tools.py
#  Tools to manage files
#
#
#  Updated by Xavier Vasques on 19/02/2015.
#

import csv
import numpy as np
from numpy import sort
import operator
import tools
import os, random

# Choose a file randomly with conditions
def random_file(dir):
    files = [os.path.join(path, filename)
             for path, dirs, files in os.walk(dir)
             for filename in files
             if filename.endswith(".csv")
             if os.path.getsize(dir+"/"+filename) > 0
             if not filename.startswith('.')]
    return random.choice(files)

#replace tab separated csv files by coma csv files
def separate_coma(input,output):
    for root, dirs, files in os.walk(input):
        for i in files:
            if not i.startswith('.'):
                input_i=input+i
                output_i=output+i
                file = open(output_i, "w")
                writer=csv.writer(file, lineterminator=',')
                lines=tools.file_lines(input_i)+1
                ncol=tools.file_col(input_i)
                for i in xrange(lines):
                    for j in xrange(ncol):
                        file.write("%s,"%tools.read_csv_tab(input_i,j,i))
                    file.write("\n")
                file.close()

# Number of columns in the csv separated by tab
def file_col(morphology_file_path):
    file = open(morphology_file_path, "rt")
    reader=csv.reader(open(morphology_file_path, "rt"), delimiter='\t')
    ncol=len(next(reader))
    return ncol

# Number of columns in the csv separated by coma
def file_col_coma(morphology_file_path):
    file = open(morphology_file_path, "rt")
    reader=csv.reader(open(morphology_file_path, "rt"), delimiter=',')
    ncol=len(next(reader))
    return ncol

# Number of lines in the csv separated by tab
def file_lines(morphology_file_path):
    file = open(morphology_file_path, "rt")
    reader=csv.reader(open(morphology_file_path, "rt"), delimiter='\t')
    for row in reader: row
    lines=reader.line_num-1
    return lines

# Number of lines in the csv separated by coma
def file_lines_coma(morphology_file_path):
    file = open(morphology_file_path, "rt")
    reader=csv.reader(open(morphology_file_path, "rt"), delimiter=',')
    for row in reader: row
    lines=reader.line_num-1
    return lines

###################    READ CSV     ###################################

# READ TEXT
def read_csv_tab(morphology_file_path,col,line):
    with open(morphology_file_path,'rb') as f:
        rows=list(csv.reader(f,delimiter='\t'))
        b=rows[line][col]
    return b

def read_csv_coma(morphology_file_path,col,line):
    with open(morphology_file_path,'rb') as f:
        rows=list(csv.reader(f,delimiter=','))
        b=rows[line][col]
    return b

# READ FLOAT

# Read the .csv files and put data into a matrice separated by tab
def read_float_tab(morphology_file_path,col,row):
    col_num=col
    row_num=row
    ncol=file_col(morphology_file_path) # Number of columns in the csv
    lines=file_lines(morphology_file_path) # Number of lines in the csv
    if lines==1: # If the number of lines are 1 (excluding hearders) special treatement
        data = np.genfromtxt(morphology_file_path, delimiter='\t', names=True, dtype=float,usecols=col_num)
        data.dtype = np.dtype(float)
        return data
    else: # Initiate matrice and put data into the matrice
        a=[[0 for x in xrange(lines)] for x in xrange(ncol)]
        data = np.genfromtxt(morphology_file_path, delimiter='\t', names=True, dtype=float,usecols=col_num)
        data.dtype = np.dtype(float)
        a[col_num][row_num]=data[(row_num)]
    return a[col_num][row_num]

def read_float_coma(morphology_file_path,col,row):
    col_num=col
    row_num=row
    ncol=file_col_coma(morphology_file_path) # Number of columns in the csv
    lines=file_lines_coma(morphology_file_path) # Number of lines in the csv
    if lines==1: # If the number of lines are 1 (excluding hearders) special treatement
        data = np.genfromtxt(morphology_file_path, delimiter=',', names=True, dtype=None,usecols=col_num)
        data.dtype = np.dtype(float)
        return data
    else: # Initiate matrice and put data into the matrice
        a=[[0 for x in xrange(lines)] for x in xrange(ncol)]
        data = np.genfromtxt(morphology_file_path, delimiter=',', names=True, dtype=float,usecols=col_num)
        data.dtype = np.dtype(float)
        a[col_num][row_num]=data[(row_num)]
    return a[col_num][row_num]

# Read the .csv files and put text data into a matrice separated by tab
def morphology_matrice_txt(morphology_file_path,col,row):
    col_num=col
    row_num=row
    ncol=file_col(morphology_file_path) # Number of columns in the csv
    lines=file_lines(morphology_file_path) # Number of lines in the csv
    if lines==1: # If the number of lines are 1 (excluding hearders) special treatement
        data = np.genfromtxt(morphology_file_path, delimiter='\t', names=True, dtype=None,usecols=col_num)
        #data.dtype = np.dtype(float)
        return data
    else: # Initiate matrice and put data into the matrice
        a=[[0 for x in xrange(lines)] for x in xrange(ncol)]
        data = np.genfromtxt(morphology_file_path, delimiter='\t', names=True, dtype=None,usecols=col_num)
        #data.dtype = np.dtype(float)
        a[col_num][row_num]=data[(row_num)]
    return a[col_num][row_num]

# Read the .csv files and put texte data into a matrice separated by coma
def morphology_matrice_txt_coma(morphology_file_path,col,row):
    col_num=col
    row_num=row
    ncol=file_col(morphology_file_path) # Number of columns in the csv
    lines=file_lines(morphology_file_path) # Number of lines in the csv
    if lines==1: # If the number of lines are 1 (excluding hearders) special treatement
        data = np.genfromtxt(morphology_file_path, delimiter=',', names=True, usecols=col_num)
        #data.dtype = np.dtype(float)
        return data
    else: # Initiate matrice and put data into the matrice
        a=[[0 for x in xrange(lines)] for x in xrange(ncol)]
        data = np.genfromtxt(morphology_file_path, delimiter=',', names=True, usecols=col_num)
        #data.dtype = np.dtype(float)
        a[col_num][row_num]=data[(row_num)]
    return a[col_num][row_num]

# Read the .csv files and put texte data into a matrice separated by coma
def morphology_matrice_txt_coma_with_title(morphology_file_path,col,row):
    col_num=col
    row_num=row
    ncol=file_col(morphology_file_path) # Number of columns in the csv
    lines=file_lines(morphology_file_path)+1 # Number of lines in the csv
    if lines==1: # If the number of lines are 1 (excluding hearders) special treatement
        data = np.genfromtxt(morphology_file_path, delimiter=',', dtype=None,usecols=col_num)
        #data.dtype = np.dtype(float)
        return data
    else: # Initiate matrice and put data into the matrice
        a=[[0 for x in xrange(lines)] for x in xrange(ncol)]
        data = np.genfromtxt(morphology_file_path, delimiter=',', dtype=None,usecols=col_num)
        #data.dtype = np.dtype(float)
        a[col_num][row_num]=data[(row_num)]
    return a[col_num][row_num]

def morphology_matrice_coma_with_title(morphology_file_path,col,row):
    col_num=col
    row_num=row
    ncol=file_col_coma(morphology_file_path) # Number of columns in the csv
    lines=file_lines_coma(morphology_file_path)+1 # Number of lines in the csv
    if lines==1: # If the number of lines are 1 (excluding hearders) special treatement
        data = np.genfromtxt(morphology_file_path, delimiter=',', dtype=None,usecols=col_num)
        data.dtype = np.dtype(float)
        return data
    else: # Initiate matrice and put data into the matrice
        a=[[0 for x in xrange(lines)] for x in xrange(ncol)]
        data = np.genfromtxt(morphology_file_path, delimiter=',', dtype=float,usecols=col_num)
        data.dtype = np.dtype(float)
        a[col_num][row_num]=data[(row_num)]
    return a[col_num][row_num]



# Read the .csv files and put data into a matrice separated by  coma
def morphology_matrice_coma(morphology_file_path,col,row):
    col_num=col
    row_num=row
    ncol=file_col_coma(morphology_file_path) # Number of columns in the csv
    lines=file_lines_coma(morphology_file_path) # Number of lines in the csv
    if lines==1: # If the number of lines are 1 (excluding hearders) special treatement
        data = np.genfromtxt(morphology_file_path, delimiter=',', names=True, dtype=None,usecols=col_num)
        data.dtype = np.dtype(float)
        return data
    else: # Initiate matrice and put data into the matrice
        a=[[0 for x in xrange(lines)] for x in xrange(ncol)]
        data = np.genfromtxt(morphology_file_path, delimiter=',', names=True, dtype=float,usecols=col_num)
        data.dtype = np.dtype(float)
        a[col_num][row_num]=data[(row_num)]
    return a[col_num][row_num]


# Sort values within the CSV
def sortcsvbymanyfields(csvfilename, themanyfieldscolumnnumbers):
    with open(csvfilename, 'rb') as f:
        readit = csv.reader(f)
        thedata = list(readit)
    thedata.sort(key=operator.itemgetter(themanyfieldscolumnnumbers))
    with open(csvfilename, 'wb') as f:
        writeit = csv.writer(f)
        writeit.writerows(thedata)
