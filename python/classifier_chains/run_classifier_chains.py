import getopt
import sys
import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.covariance import EmpiricalCovariance
from classifier_chain import ClassifierChain

def do_cross_validation(X, Y):
    X, Y = shuffle(X, Y)
    folds = 10
    kf = KFold(n_splits=folds, shuffle=True)
    cc = ClassifierChain()
    zero_one_score = 0
    hamming_score = 0
    accuracy = 0

    for train, test in kf.split(X):
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        cc.fit(X_train, y_train)
        zero_one_score += cc.zero_one_loss_score(X_test, y_test)
        hamming_score += cc.hamming_loss_score(X_test, y_test)
        accuracy += cc.accuracy_score(X_test, y_test)

    zero_one_score /= folds
    hamming_score /= folds
    accuracy /= folds

    #print "0/1 Loss:", zero_one_score
    #print "Hamming Loss:", hamming_score
    #print "Accuracy:", accuracy
    #print

    return [zero_one_score, hamming_score, accuracy]

def get_covariance(X):
    cov = EmpiricalCovariance(assume_centered=True)
    cov.fit(X)
    return cov.covariance_

def get_label_order(cov_matrix):
    np.fill_diagonal(cov_matrix, 0) # ignore the diagonal values
    num_cols = cov_matrix.shape[1]
    num_rows = cov_matrix.shape[0]
    zero_column = np.zeros((num_rows,))
    zero_row = np.zeros((1, num_cols))
    new_label_order = range(num_cols)
    index = 0
    
    while index < (num_cols - 2):
        argmax = np.argmax(cov_matrix)
        
        row_index = argmax / num_cols
        col_index = argmax % num_cols

        print "row_index:", row_index
        print "col_index:", col_index
        
        cov_matrix[:,col_index] = zero_column
        cov_matrix[col_index,:] = zero_row
        cov_matrix[:,row_index] = zero_column
        cov_matrix[row_index,:] = zero_row

        print cov_matrix

        #new_label_order.append(row_index)
        #new_label_order.append(col_index)

        temp  = new_label_order[index]
        swap_index = new_label_order.index(col_index)
        new_label_order[index] = col_index
        new_label_order[swap_index] = temp

        print "Col_index swap:", new_label_order

        temp = new_label_order[index + 1]
        swap_index = new_label_order.index(row_index)
        new_label_order[index + 1] = row_index
        new_label_order[swap_index] = temp

        index += 2
        print new_label_order

    return new_label_order


def get_data(filename, num_labels):
    dataset = pd.read_csv(filename, delimiter=',')
    X = dataset.iloc[:,0:-num_labels].as_matrix()
    Y = dataset.iloc[:,-num_labels:].as_matrix()
    return X, Y

def usage():
    print 'Usage: python ' + sys.argv[0] + ' -d <path to dataset> -l <number of labels>'

def main():
    data_file = None
    num_labels = 0

    try:
		options, remainder = getopt.getopt(sys.argv[1:], 'd:l:', ['data=', 'labels='])
    except getopt.GetoptError as err:
        print err
        usage()
        sys.exit(2)
	
    for opt, arg in options:
        if opt in ('-d', '--data'):
            data_file = arg
        elif opt in ('-l', '--labels'):
            num_labels = int(arg)

	X, Y = get_data(data_file, num_labels)
    cov = get_covariance(Y)
    print cov
    order = get_label_order(cov)
    test_Y = np.zeros_like(Y)
    for i, column in enumerate(order):
        test_Y[:,i] = Y[:,column]

    with open('plain_ol_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['0/1 loss', 'hamming', 'accuracy'])

        for i in range(50):
            #print "PLAIN OLD Y:"
            writer.writerow(do_cross_validation(X, Y))
        
    with open('experiment_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['0/1 loss', 'hamming', 'accuracy'])

        for i in range(50):
            #print "TEST Y"
            writer.writerow(do_cross_validation(X, test_Y))
    """
    for i in range(3):
        print "RANDOM SHUFFLE Y"
        np.random.shuffle(Y.T)
        do_cross_validation(X, Y)
    """

if __name__ == '__main__':
    main()