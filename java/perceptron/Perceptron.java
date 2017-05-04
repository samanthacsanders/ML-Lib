package perceptron;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import toolkit.Matrix;

public class Perceptron extends toolkit.SupervisedLearner {
	
	private double lr = 0.001; // learning rate
	private Perceptronette[] perceptron;
	Random rand;
	private int uniqueLabels = 0;

	public Perceptron(Random r) {
		rand = r;
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// I'm doing this the 1-output-node-per-label way
		// The number of unique labels
		uniqueLabels = labels.valueCount(0);
		
		// For each label initiate a perceptron-ette (if there are more than 2 labels)
		if (uniqueLabels > 2) {
//			System.out.println("UNIQUE LABELS: " + uniqueLabels);
			// As many perceptrons as there are labels
			perceptron = new Perceptronette[uniqueLabels];
			// Make one set of labels for each class
			Matrix[] newLabels = divideDataset(labels);
			
//			for (int i = 0; i < newLabels.length; i++) {
//				newLabels[i].print();
//			}
			
			// Initialize all of the perceptrons
			for (int i = 0; i < perceptron.length; i++) {
				perceptron[i] = new Perceptronette(features, newLabels[i], rand, lr);
			}
			// Train the perceptrons
			for (int i = 0; i < perceptron.length; i++) {
//				System.out.println("TRAINING PERCEPTRON " + i);
				perceptron[i].train();
			}
			
		} else {
			
//			System.out.println("TWO CLASSES");
			
			// If there are only two labels then we only need to train one perceptron
			perceptron = new Perceptronette[1];
//			labels.print();
			perceptron[0] = new Perceptronette(features, labels, rand, lr);
			perceptron[0].train();
		}
		
	}
	
	// Create a new set of labels for each perceptron (unless there are only two labels)
	private Matrix[] divideDataset(Matrix labels) {
		Matrix[] newLabels = new Matrix[uniqueLabels];
		
		// initate each new labels matrix
		for (int i = 0; i < newLabels.length; i++) {
			newLabels[i] = new Matrix();
			newLabels[i].setBinaryLabelMatrix(labels.rows(), labels.cols());
			
		}

		for (int row = 0; row < labels.rows(); row++) {
			int classIndex = (int)labels.get(row, 0); // range: [0, uniqueLabels-1]

			for (int datasetIndex = 0; datasetIndex < uniqueLabels; datasetIndex++) {
				if (datasetIndex == classIndex) {
					newLabels[datasetIndex].set(row, 0, 1); // set the label to 1
				} else {
					newLabels[datasetIndex].set(row, 0, 0);
				}
			}
		}
		
//		for (int i = 0; i < newLabels.length; i++) {
//			newLabels[i].print();
//		}
		
		return newLabels;
	}
	
	@Override
	public void predict(double[] features, double[] prediction) throws Exception {
		double pred = 0;
		
		// Print features
//		System.out.println("Features for prediction");
//		for (double feature : features) {
//			System.out.print(feature + "\t");
//		}
//		System.out.println();
		
		
		if (perceptron.length == 1) {
			pred = perceptron[0].predict(features).output;
		} else {
			Prediction[] votes = new Prediction[perceptron.length];
			for (int i = 0; i < perceptron.length; i++) {
				votes[i] = perceptron[i].predict(features);
//				System.out.println("votes[" + i + "]: " + votes[i].output + "  net: " + votes[i].net);
			}
	
			pred = (double)getVote(votes);
			
		}
		
		prediction[0] = pred;
//		System.out.println("PREDICTION: " + prediction[0]);
		
	}
	
	// returns the class with the highest net value
	private int getVote(Prediction[] votes) {
		int vote = 0;
		double maxNet = Double.MIN_VALUE;
		
		for (int i = 0; i < votes.length; i++) {
			 if (votes[i].net > maxNet) {
				 vote = i;
				 maxNet = votes[i].net;
//				 System.out.println("maxNet " + i + ": " + maxNet + "   vote: " + vote);
			 }
		}
//		System.out.println("VOTE: " + vote);
		return vote;
	}
	
	private class Perceptronette {
		float[] weights;
		Matrix features, labels;
		Random rand;
		double lr; // learning rate
		double zeroThreshold = 10E-5;	
		
		public Perceptronette(Matrix features, Matrix labels, Random rand, double lr) {
			this.features = features;
			this.labels = labels;
			this.rand = rand;
			this.lr = lr;
			// Initialize weights to 0
			weights = new float[features.cols() + 1]; //augmented weight matrix
		}
		
		public void train() {
			
			
			
//			float[] initialWeights = new float[weights.length];
			int epoch = 0;
//			boolean keepGoing = true;
//			boolean weightsChanged = false;
			int staySameCount = 0;
			int maxStaySameCount = 10;
//			double prevAccuracy = 0.01;
			double accuracy = 0.1;
			double squaredError = 0;
			double newSquaredError = 0;
			
			FileWriter writer;
			
			try {
				writer = new FileWriter("/home/samanthasanders/Documents/ML_Toolkit/ls_lr_0.001.csv");
			
			
				do {
					epoch++;
	//				weightsChanged = false;
					newSquaredError = 0;
					features.shuffle(rand, labels); // shuffle the examples between epochs
	//				features.print();
				
					// create deep copy of weights array
	//				initialWeights = Arrays.copyOf(weights, weights.length);
					
					// loop through all of the examples
					for (int row = 0; row < features.rows(); row++) {
						//calculate net
						float net = calcNet(row);
						
						// calculate output
						int output = 0;
						if (net > 0) {
							output = 1;
						}
						// stochastic approach (instead of batch)
	//					weightsChanged = 
						updateWeights(output, row);
	//					if (weightsChanged) {
	//						staySameCount = 0;
	//					} else {
	//						staySameCount++;
	//					}
	//					
						newSquaredError = calcSE(output, row, newSquaredError);
						
						
//						try {
	//						prevAccuracy = accuracy;

							
	//						double accDiff = accuracy - prevAccuracy;
	//						if ( accDiff <= 0.01 && accDiff >= 0) {
	//							staySameCount++;
	//						} else {
	//							staySameCount = 0;
	//						}
	//						System.out.println("Accuracy: " + accuracy);
//						} catch (Exception e) {
//							// TODO Auto-generated catch block
//							e.printStackTrace();
//						}
						// check if weights have changed
	//					if (!keepGoing && compare(weights, initialWeights) != 0) keepGoing = true;
					}
					
					if (Math.abs(newSquaredError - squaredError) < 3) {
						staySameCount++;
					} else {
						staySameCount = 0;
					}
					
					squaredError = newSquaredError;
					try {
						accuracy = measureAccuracy(features, labels, new Matrix());
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
					writer.append(Integer.toString(epoch));
					writer.append(',');
					writer.append(Double.toString(accuracy));
					writer.append('\n');
//					System.out.println("squaredError: " + squaredError);

				} while (staySameCount < maxStaySameCount && epoch < 10000);
				
//				System.out.println("epoch " + epoch);
				writer.append("Final Weights");
				writer.append('\n');
				for (int i = 0; i < weights.length; i++) {
					System.out.println(weights[i]);
					writer.append(Float.toString(weights[i]));
					writer.append(',');
				}
				
				
				

				writer.flush();
				writer.close();
				
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			
			
		}
		
		public Prediction predict(double[] features) {
			double net = 0;
			int output = 0;
			
			for (int i = 0; i < features.length; i++) {
				net += features[i] * weights[i];
			}
			net += weights[weights.length - 1];
			if (Math.abs(net) < zeroThreshold) net = 0;
			if(net > 0) output = 1;
			
			return new Prediction(output, net);
		}
		
		private float calcNet(int row) {
			float net = 0;
//			System.out.print("net row " + row + " = ");
			for (int col = 0; col < features.cols(); col++) {			
//				System.out.print(features.get(row, col) + " * " + weights[col] + " + ");
				net += features.get(row, col) * weights[col];
			}
			net += weights[weights.length - 1]; // bias
			
			if (Math.abs(net) < zeroThreshold) {
				net = 0;
			}
//			System.out.println(weights[weights.length - 1] + " = " + net);
			return net;
		}
		
		private boolean updateWeights(int output, int row) {
			double[] deltaW = new double[weights.length];
			int target = (int)labels.get(row, 0);
			boolean weightsChanged = false;
			
			for (int col = 0; col < features.cols(); col++) {
				double x_i = features.get(row, col);
				deltaW[col] += lr * (target - output) * x_i;
				
			}
			// don't forget the bias!
			deltaW[deltaW.length - 1] = lr * (target - output);

			for (int i = 0; i < weights.length; i++) {
				weights[i] += deltaW[i];
				if (Math.abs(deltaW[i]) > zeroThreshold) {
					weightsChanged = true;
				}
				if (Math.abs(weights[i]) < zeroThreshold) {
					weights[i] = 0;
				}
			}
			
			return weightsChanged;
		}
		
		private double calcSE(int output, int row, double MSE) {
			int target = (int)labels.get(row, 0);
			MSE += Math.pow((target - output), 2);
			return MSE;
		}
		
		// compares two arrays. If different return -1, if same return 0
		private int compare(float[] array1, float[] array2) {
			if (array1.length != array2.length) return -1;
			for (int i = 0; i < array1.length; i++) {
				if (array1[i] != array2[i]) return -1;
			}
			return 0;
		}
	}
}