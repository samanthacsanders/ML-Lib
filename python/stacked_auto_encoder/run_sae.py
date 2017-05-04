import sys
import getopt
import pandas as pd
from sklearn.model_selection import train_test_split
from StackedAutoEncoder import StackedAutoEncoder

def get_data(input_file):
	"""Takes in a .csv file and returns two numpy arrays: data and labels"""
	dataset = pd.read_csv(input_file, delimiter=',')
	X = dataset.iloc[:,:-1].as_matrix()
	y = dataset.iloc[:,-1].as_matrix()
	return X, y

def run_stacked_auto_encoder(data, labels):
    # NOTE: Make sure this is correct!
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=43)
    print data.shape[0]

    sae = StackedAutoEncoder()
    sae.fit(X_train, y_train)
    #print sae.predict(X_test)
    print sae.score(X_test, y_test)

def main():
	try:
		options, remainder = getopt.getopt(sys.argv[1:], 'd:', ['dataset='])
	except getopt.error, msg:
		print msg
	
	for opt, arg in options:
		if opt in ('-d', '--dataset'):
			dataset_filename = arg
	
	data, labels = get_data(dataset_filename)
	run_stacked_auto_encoder(data, labels)

if __name__ == '__main__':
	main()