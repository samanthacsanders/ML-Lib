import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

class ClassifierChain():

    def __init__(self):
        self.h = [] # list of base classifiers
        self.num_labels = None

    def fit(self, X, Y):
        """
        training phase for training set D
        @param X ndarray Data
        @param y ndarray Labels
        @return h [] Trained base models (SVC)
        """
        self.num_labels = Y.shape[1]
        #self.h = [SVC() for label in range(self.num_labels)]
        self.h = [DecisionTreeClassifier() for label in range(self.num_labels)]
        #self.h = [MLPClassifier() for label in range(self.num_labels)]
        #self.h = [KNeighborsClassifier() for label in range(self.num_labels)]

        for label in range(self.num_labels):
            # binary transformation
            D = np.concatenate((X, Y[:,0:label]), axis=1)
            y = Y[:,label]

            self.h[label].fit(D, y)

    def predict(self, X):
        """
        prediction phase for a test instance x
        @param X ndarray The instances to classify
        @param num_labels int Number of labels to predict
        @param h [] Trained classifiers for each label
        @return y_prime ndarray Predicted labels 
        """
        X_prime = X
        y = self.predict_n_reshape(X_prime, self.h[0])
        y_prime = y

        for label in range(1, self.num_labels):

            X_prime = np.concatenate((X_prime, y), axis=1)
            y = self.predict_n_reshape(X_prime, self.h[label])    
            y_prime = np.concatenate((y_prime, y), axis=1)

        return y_prime

    def predict_n_reshape(self, X, h):
        """
        Predicts and reshapes to 2d numpy array
        @param X ndarray Data
        @param h Classifier Trained classifier
        """
        y = h.predict(X)
        return np.atleast_2d(y).T # convert to 2d array

    def zero_one_loss_score(self, X, y):
        """
        Calculate the 0/1 loss score
        @param X ndarray Test samples
        @param y ndarray True labels for X
        @return score float 0/1 loss of self.predict(X) wrt y
        """
        N = float(X.shape[0]) # number of examples
        y_predict = self.predict(X)
        sums = np.sum(np.all(y == y_predict, axis=1))
        return 1 - (1 / N) * sums

    def hamming_loss_score(self, X, y):
        """
        Calculate the Hamming loss score
        @param X ndarray Test samples
        @param y ndarray True labels for X
        @return score float Hamming loss of self.predict(X) wrt y
        """
        N = float(X.shape[0]) # number of examples
        L = float(y.shape[1]) # number of labels
        y_predict = self.predict(X)
        sums = np.sum(y == y_predict)
        return 1 - (1 / (N * L)) * sums

    def accuracy_score(self, X, y):
        """
        Calculate the accuracy
        @param X ndarray Test samples
        @param y ndarray True labels for X
        @return score float accuracy of self.predict(X) wrt y
        """
        N = float(X.shape[0]) # number of examples
        y_predict = self.predict(X)

        ands = np.sum(np.logical_and(y, y_predict), axis=1)
        ors = np.sum(np.logical_or(y, y_predict), axis=1).astype(float, copy=False)

        sums = np.sum(ands / ors)

        return (1 / N) * sums