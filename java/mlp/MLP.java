package mlp;

import java.util.Random;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class MLP extends SupervisedLearner{
	Random rand;
	private double[][] weights, errors, outputs;
	private int[] layerNodeCount = {8, -1}; // the -1 is the output layer, we will initialize this later
	private double lr = 0.1; // learning rate
	private Matrix trainF, trainL, valF, valL;
	private double[][] bssfWeights;
	private int stopCount = 0, maxStopCount = 5;
	private double bssfMSE = Double.MAX_VALUE;
	
	private boolean test = false;
	
	public NewMLP(Random rand) {
		this.rand = rand;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
		
		if (test) {
			initializeTestNetwork();
			feedforward(features.row(0), weights);
			System.out.println("Outputs");
			print2DArray(outputs);
			propagateBack(features.row(0), labels.row(0));
			
			
			
		} else {
			initializeNetwork(features.cols(), labels.valueCount(0));
			
			System.out.println("Weights");
			print2DArray(weights);
			
			System.out.println();
			
			System.out.println("Outputs");
			print2DArray(outputs);
			
			System.out.println();
			
			System.out.println("Errors");
			print2DArray(errors);
			bssfWeights = copy2dArray(weights);
			features.normalize();
			features.shuffle(rand, labels);
			splitData(0.5, features, labels);	
			
			do {
				trainF.shuffle(rand, trainL);
				// an epoch
				for (int row = 0; row < trainF.rows(); row++) {
					feedforward(trainF.row(row), weights);
					propagateBack(trainF.row(row), trainL.row(row));
				}
				
			} while(checkStoppingCriteria());
		}		
	}

	@Override
	public void predict(double[] features, double[] prediction) throws Exception {
		double[] normFeatures = getNormalizedFeatures(features);
		feedforward(normFeatures, bssfWeights);
		prediction[0] = getMaxOutputIndex();
//		System.out.println("Weights");
//		print2DArray(weights);
//		System.out.println("bssfWeights");
//		print2DArray(bssfWeights);
	}
	
	private int getMaxOutputIndex() {
		double maxVal = Double.MIN_VALUE;
		int index = 0;
		
		for (int k = 0; k < outputs[outputs.length - 1].length; k++) {
			if (outputs[outputs.length - 1][k] > maxVal) {
				maxVal = outputs[outputs.length - 1][k];
				index = k;
			}
		}
		return index;
	}
	
	// Normalizes the features in accordance with the training set min/max
	private double[] getNormalizedFeatures(double[] features) {
		double[] normFeatures = new double[features.length];
		
		for (int col = 0; col < features.length; col++) {
			double colMin = trainF.columnMin(col);
			double colMax = trainF.columnMax(col);
			normFeatures[col] = (features[col] - colMin) / (colMax - colMin);
		}
		return normFeatures;
	}
	

	
	private void feedforward(double[] features, double[][] weights) {
		double[] input = new double[features.length + 1];
		System.arraycopy(features, 0, input, 0, features.length);
		input[input.length - 1] = 1.0; // bias
		
		// loop through the layers
		for (int layer = 0; layer < layerNodeCount.length; layer++) {
			
			// print inputs
//			System.out.println("Inputs");
//			for (int k = 0; k < input.length; k++) {
//				System.out.print(input[k] + "  ");
//			}
//			System.out.println();
			
			// loop through the nodes in this layer
			for (int node = 0; node < layerNodeCount[layer]; node++) { // +1 for bias
				double net = 0;
//				System.out.print("net = ");
				// loop through the inputs to this node and calculate the net value
				for (int i = 0; i < input.length; i++) {
					int weightIDx = node * input.length + i;
//					System.out.print(weights[layer][weightIDx] + " * " + input[i] + " + ");
					net += weights[layer][weightIDx] * input[i];
				}
//				System.out.println("= " + net);
				outputs[layer][node] = sigmoid(net);
			}
			input = new double[outputs[layer].length + 1]; // bias
			System.arraycopy(outputs[layer], 0, input, 0, outputs[layer].length);
			input[input.length - 1] = 1.0; // bias
		}
	}
	
	private void propagateBack(double[] features, double[] labels) {
		calculateOutputError(labels[0]);
		calculateHiddenError();
		
//		System.out.println("errors");
//		print2DArray(errors);
		
		updateWeights(features);
	}
	
	private void calculateOutputError(double label) {
		double target = 0, output = 0;
		// loop through the output nodes
		for (int outNode = 0; outNode < outputs[outputs.length - 1].length; outNode++) {
			target = 0;
			if (outNode == (int)label) target = 1;
			output = outputs[outputs.length - 1][outNode];
			errors[errors.length - 1][outNode] = output * (1 - output) * (target - output);
		}
	}
	
	private void calculateHiddenError() {
		// loop through the hidden layers
		for (int layer = layerNodeCount.length - 2; layer >= 0; layer--) {
//			System.out.println("Layer: " + layer);
			
			// loop through the nodes in this hidden layer
			for (int node = 0; node < layerNodeCount[layer]; node++) {
				// this node's output
				double output = outputs[layer][node];
				double sum = 0;
//				System.out.print("sum = ");
				
				// loop through the nodes in the layer in front of this one
				for (int k = 0; k < layerNodeCount[layer+1]; k++) {
					// get this (k) node's error
					double err = errors[layer+1][k];
					
					int weightIDx = node + k * (layerNodeCount[layer + 1]); // don't forget the bias
//					System.out.println("layer + 1: " + (layer+1) + "   weightIDx: " + weightIDx);
//					System.out.println("weightIDX: " + weightIDx + "   weight: " + weights[layer + 1][weightIDx]);
//					System.out.print(err + " * " + weights[layer + 1][weightIDx] + " + ");
					sum += err * weights[layer + 1][weightIDx];
				}
//				System.out.println(" = " + sum);
				errors[layer][node] = output * (1 - output) * sum;
			}
		}
	}
	
	private void updateWeights(double[] features) {
		double[] output = new double[features.length + 1];
		System.arraycopy(features, 0, output, 0, features.length);
		output[output.length - 1] = 1.0; // bias
		
		// loop through all of the layers
		for (int layer = 0; layer < layerNodeCount.length; layer++) {
			
			// loop through all of the nodes in this layer
			for (int node = 0; node < layerNodeCount[layer]; node++) {
				double err = errors[layer][node];
				
				// calculate the deltaWs from the nodes in layer 'layer-1' to this node
				// the 'output' comes from layer l-1
				for (int out = 0; out < output.length; out++) {
					
					double deltaW = lr * err * output[out];
//					System.out.println("layer " + layer + " node " + node + " deltaW = " + lr + " * " + err + " * " + output[out] + " = " + deltaW);
					int weightIDx = node * output.length + out;
					weights[layer][weightIDx] += deltaW;
				}
			}
			// get the new outputs
			output = new double[outputs[layer].length + 1];
			System.arraycopy(outputs[layer], 0, output, 0, outputs[layer].length);
			output[output.length - 1] = 1.0; // bias
		}
		
	}
	
	private double sigmoid(double net) {
		return 1 / (1 + Math.exp(-1 * net));
	}
	
	private void initializeTestNetwork() {
		weights = new double[layerNodeCount.length][6];
		layerNodeCount[layerNodeCount.length - 1] = 2;
		
		weights[0][0] = 0.2;
		weights[0][1] = -0.1;
		weights[0][2] = 0.1;
		weights[0][3] = 0.3;
		weights[0][4] = -0.3;
		weights[0][5] = -0.2;
		
		weights[1][0] = -0.2;
		weights[1][1] = -0.3;
		weights[1][2] = 0.1;
		weights[1][3] = -0.1;
		weights[1][4] = 0.3;
		weights[1][5] = 0.2;
		
		outputs = new double[layerNodeCount.length][2];
		errors = new double[layerNodeCount.length][2];
	}
	
	private void initializeNetwork(int numFeatures, int numLabels) {
		layerNodeCount[layerNodeCount.length - 1] = numLabels;
		initializeWeights(numFeatures);
		initializeErrors();
		initializeOutputs();
	}
	
	private void initializeWeights(int numFeatures) {
		weights = new double[layerNodeCount.length][];
		for (int layer = 0; layer < layerNodeCount.length; layer++) {
			int numWeights;
			if (layer == 0) {
				System.out.println("numFeatures: " + numFeatures + "   layerNodeCount[" + layer + "]: " + layerNodeCount[layer]); 
				numWeights = (numFeatures + 1) * layerNodeCount[layer]; // +1 for bias
			} else {
				numWeights = (layerNodeCount[layer-1] + 1) * layerNodeCount[layer]; // +1 for bias
			}
			System.out.println("layer " + layer + " numWeights: " + numWeights);
			weights[layer] = new double[numWeights];
			for (int node = 0; node < weights[layer].length; node++) {
//				weights[layer][node] = (rand.nextGaussian() - 0.5) / 10;
				weights[layer][node] = rand.nextGaussian();
			}
		}
	}
	
	private void initializeErrors() {
		errors = new double[layerNodeCount.length][];
		for (int layer = 0; layer < layerNodeCount.length; layer++) {
			errors[layer] = new double[layerNodeCount[layer]]; // the bias doesn't have an error so no +1
		}
	}
	
	private void initializeOutputs() {
		outputs = new double[layerNodeCount.length][];
		for (int layer = 0; layer < layerNodeCount.length; layer++) {
			outputs[layer] = new double[layerNodeCount[layer]]; // no bias in the output layer
		}
	}
	
	private void splitData(double percentTrain, Matrix features, Matrix labels) {
		int trainingRows = (int) ((double)features.rows() * percentTrain);

		trainF = new Matrix(features, 0, 0, trainingRows, features.cols());
		valF = new Matrix(features, trainingRows, 0, features.rows() - trainingRows, features.cols());
		trainL = new Matrix(labels, 0, 0, trainingRows, labels.cols());
		valL = new Matrix(labels, trainingRows, 0, labels.rows() - trainingRows, labels.cols());
	}
	
	private boolean checkStoppingCriteria() {	
		double mse = calcMSE(valF, valL);
		
		System.out.println("MSE: " + mse);
		System.out.println("bssfMSE: " + bssfMSE + "\n");
		
		if (mse < bssfMSE) {
			bssfMSE = mse;		
			bssfWeights = copy2dArray(weights);
			stopCount = 0;
		} else {
			stopCount++;
		}
		
		if (stopCount >= maxStopCount) return false;
		return true;	
	}
	
	private double calcMSE(Matrix features, Matrix labels) {
		double sse = 0;
		
		for (int row = 0; row < features.rows(); row++) {
			double[] example = features.row(row);
			double label = labels.row(row)[0];
			int classPrediction = validationPredict(example);
			System.out.println("label: " + label + "  prediction: " + classPrediction);
			if (classPrediction != label) sse += 1;
		}
		return sse / features.rows();
	}
	
	// this assumes the features are already normalized
	private int validationPredict(double[] features) {
		feedforward(features, weights);
		int prediction = getMaxOutputIndex();
		return prediction;
	}
	
	private double[][] copy2dArray(double[][] array) {
		double[][] copy = new double[array.length][];
		for (int i = 0; i < array.length; i++) {
			double[] aCopy = array[i];
			int aLength = aCopy.length;
			copy[i] = new double[aLength];
			System.arraycopy(aCopy, 0, copy[i], 0, aLength);
		}
		return copy;
	}
	
	private void print2DArray(double[][] array) {
		for (int row = 0; row < array.length; row++) {
			for (int col = 0; col < array[row].length; col++) {
				System.out.print("[" + array[row][col] +"] ");
			}
			System.out.println();
		}
		System.out.println();
	}

}
