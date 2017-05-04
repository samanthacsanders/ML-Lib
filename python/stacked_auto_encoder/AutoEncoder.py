import numpy as np

class AutoEncoder():

    def __init__(self, hidden_layer_size=2):
        self.learning_rate = 0.01
        self.hidden_layer = hidden_layer_size
        self.tol = 0.1
        self.max_same_iter = 200
        self.coefs_ = []

    def fit(self, X):
        """
        Trains an auto encoder on data set X
        @param X ndarray of data (input and output) no bias here
        @return [ndarray] trained weight matrices
        """
        weights = self.init_weights(X.shape[1], self.hidden_layer, X.shape[1])

        # Testing
        #init_weights1 = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, -0.4], [0.1, 0.1]])
        #init_weights2 = np.array([[0.2, 0.1, 0.3], [-0.1, 0.2, -0.2], [0.4, 0.4, 0.4]])
        #weights = [init_weights1, init_weights2]

        self.coefs_ = self.backprop(X, weights)

    def get_coefs(self):
        return self.coefs_

    def add_bias(self, X):
        """
        Adds a bias (1) column
        @param X ndarray
        @return ndarray X with an additional column of ones
        """
        X_bias = np.ones((X.shape[0], X.shape[1] + 1))
        X_bias[:,:-1] = X
        return X_bias

    def init_weights(self, input_layer, hidden_layer, output):
        """
        Initializes two layers of weights
        @param input_layer int The size of the input layer
        @param hidden_layer int The size of the hidden layer
        @param output int The size of the output layer
        @return [ndarray] The first element are weights from input_layer
            to hidden_layer and the second element are weights from
            hidden_layer to output layer
        """
        weights = []
        weights.append(0.01 * np.random.randn(input_layer + 1, hidden_layer)) # +1 for bias weights
        weights.append(0.01 * np.random.randn(hidden_layer + 1, output)) # +1 for bias weights
        return weights

    def backprop(self, X, weights):
        """
        Performs backpropogation on the MLP defined by weights.
        @param X ndarray The data instances (inputs and outputs)
        @param weights [ndarray] The initialized weight matrices
        @return [ndarray] Trained weights
        """
        bias_list_X = self.add_bias(X).tolist()
        # instances is a list of ndarrays of shape (1, n)
        instances = [np.expand_dims(np.array(xi), axis=0) for xi in bias_list_X]
        num_instances = len(instances)
        mse = 100
        iter_count = 0
        data_index = 0

        while mse > self.tol and iter_count < self.max_same_iter: # stopping criterion
            instance = instances[data_index]
            self.update_index(data_index, num_instances - 1)
            delta_ws = []

            # Calculate weight changes for output layer
            # propogate the instance forward through the network
            input_results, outputs, nets = self.forward(instance, weights)
            # calculate the output error
            final_output = outputs.pop()
            error = self.get_output_error(instance[:,:-1], final_output, nets.pop())
            # calculate weight change
            delta_w = self.get_weight_change(input_results.pop(), error)
            delta_ws.insert(0, delta_w)

            # Calculate weight changes for hidden layers
            for i, output in reversed(list(enumerate(outputs))):
                tp_weights = np.delete(weights[i+1].T, -1, axis=1) # transpose and delete bias column
                error = self.get_hidden_error(error, tp_weights, nets[i])
                delta_w = self.get_weight_change(input_results[i], error)
                delta_ws.insert(0, delta_w)

            for i, w in enumerate(weights):
                w += delta_ws[i]

            # updating stopping criterion conditions
            temp_mse = ((instance[:,:-1] - final_output)**2).mean(axis=None)
            if (temp_mse == mse):
                iter_count += 1
            else:
                iter_count = 0
            
            mse = temp_mse
        print mse

        return weights

    def get_output_error(self, target, output, net):
        """
        Calculates the output error value for backpropogation
        @param target ndarray Target values
        @param output ndarray Acutal output values
        @param net ndarray net values from the output layer
        """
        error = np.multiply((target - output), self.relu_df(net))
        return error

    def get_hidden_error(self, error, weights, net):
        """
        Calculates the hidden layer error value for backpropogation
        @param error ndarray The error from the layer above
        @param weights ndarray The weights from the layer above
        @param net ndarray The net values from the current layer
        """
        error = np.multiply(np.dot(error, weights), self.relu_df(net))
        return error

    def get_weight_change(self, inputs, error):
        """
        Calculates the weight change matrix
        @param inputs ndarray The inputs to the layer
        @param error ndarray The error of the layer
        @return ndarray Weight change matrix
        """
        delta_w = np.zeros((inputs.shape[1], error.shape[1]))
        input_list = inputs.tolist()[0]

        for index, i in enumerate(input_list):
            delta_w[index,:] = self.learning_rate * i * error

        return delta_w

    def forward(self, x, weights):
        """
        Takes an instance x and propogates it forward through the network.
        @param x ndarray a (1,n) numpy array. The instance to propogate through the network. 
            Already has the bias included.
        @param weights ndarray The weights for each layer. Bias weights included.
        @return inputs [ndarray] The values that are input at each layer
        @return outputs [ndarray] The values that are output at each layer
        @return nets [ndarray] The net values output at each layer
        """
        inputs = []
        outputs = []
        nets = []
        input_data = x

        for weight_matrix in weights:
            inputs.append(input_data)
            net = np.dot(input_data, weight_matrix) 
            nets.append(net)
            output = self.relu(net)
            outputs.append(output)
            input_data = self.add_bias(output)

        return inputs, outputs, nets

    def relu(self, net):
        return np.maximum(net, 0)

    def relu_df(self, net):
        zeroes = np.zeros_like(net)
        return np.greater(net, zeroes)

    def update_index(self, index, max_index):
        """
        Increments the index or sets to zero if the index
        is greater than max_index.
        @param index int Current index value
        @param max_index int Maximum value of index
        @return int incremented index value
        """
        if index > max_index:
            index = 0
        else:
            index += 1