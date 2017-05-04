from sklearn.neural_network import MLPClassifier
from AutoEncoder import AutoEncoder
import numpy as np


class StackedAutoEncoder():

    def __init__(self, hidden_layer_sizes=(100,75,50)):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.coefs_ = [] # the ith layer in the list represents the weight matrix corresponding to layer i
        self.intercepts_ = [] # holds the biases
        self.learning_rate = 0.01
        self.tol = 0.01
        self.max_iter = 1000
        #self.classifier = LogisticRegression(multi_class='multinomial') # softmax

    def fit(self, X, y):
        """fit the model to data matrix X and target y"""
        mlp_hl_size = 50
        num_classes = np.unique(y).size
        inputs = X
        bias_X = self.add_bias(X)

        # unsupervised training (training auto encoders)
        for i, layer_size in enumerate(self.hidden_layer_sizes):
            auto_encoder = AutoEncoder(hidden_layer_size=layer_size)
            auto_encoder.fit(inputs)
            weights = auto_encoder.get_coefs()
            self.coefs_.append(weights[0])
            inputs = self.forward(bias_X, self.coefs_) # no bias here
        
        # supervised training using MLP classifier
        mlp = MLPClassifier()
        mlp.fit(inputs, y)
        #print "MLP Score:", mlp.score(inputs, y)

        mlp_coefs = []
        for i, coefs in enumerate(mlp.coefs_):
            new_coefs = np.vstack((coefs, mlp.intercepts_[i]))
            self.coefs_.append(new_coefs)

        """
        # supervised training on the entire network
        #self.coefs = self.backprop(X, y, self.coefs_, self.hidden_layer_sizes + (mlp_hl,))
        """

    def add_bias(self, X):
        """Adds a bias (1) column"""
        X_bias = np.ones((X.shape[0], X.shape[1] + 1))
        X_bias[:,:-1] = X
        return X_bias
    
    def forward(self, X, weights):
        """
        Takes an ndarray and propogates it forward through the network.
        @param X ndarray a (m,n) numpy array. The instances to propogate through the network. 
            Already has the bias included
        @param weights [ndarray] The weights for each layer. Bias weights included.
        @return output ndarray The values that are output at the last layer
        """
        input_data = X

        for i in range(len(weights) - 1):
            net = np.dot(input_data, weights[i]) 
            output = self.relu(net)
            input_data = self.add_bias(output)

        net = np.dot(input_data, weights[-1])
        output = self.softmax(net)

        return output

    def relu(self, net):
        return np.maximum(net, 0)

    def relu_df(self, net):
        zeroes = np.zeros_like(net)
        return np.greater(net, zeroes)

    def softmax(self, net):
        exp = np.exp(net)
        return exp / np.sum(exp)

    def predict(self, X):
        """
        Predicts the label(s) for input X
        @param X ndarray The input data
        @return ndarray The predicted classes 
        """
        inputs = self.add_bias(X)
        output = self.forward(inputs, self.coefs_)
        return np.argmax(output, axis=1)

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels
        @param X ndarray The test samples
        @param y ndarray The true labels for X
        @return float mean accuracy of self.predict(X) with respect to y
        """
        prediction = self.predict(X)
        results = np.equal(prediction, y)
        return np.sum(results) / float(results.size)