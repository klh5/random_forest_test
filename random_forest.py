import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

def main():
	
	input_file = pd.read_csv("data.csv")
	train_data, test_data = train_test_split(input_file, test_size = 0.2) # 20% test data, 80% training

	train = pd.DataFrame(train_data)
	test = pd.DataFrame(test_data)

	trainArr = train.as_matrix(columns=train.columns[1:]) #training array
	trainRes = train.as_matrix(columns=train.columns[:1]) # training results

	rf = RandomForestClassifier(n_estimators=100) # initialize
	rf.fit(trainArr, trainRes) # fit the data to the algorithm

	testArr = test.as_matrix(columns=test.columns[1:])
	results = rf.predict(testArr)

	test.columns = ['class', 'pc1', 'pc2']
	
	test['predictions'] = results

	i = 0

	for index, row in test.iterrows():
		if row['class'] == row['predictions']:
			i += 1

	num_correct = float(i) / float(len(test.index))
	percentage = num_correct * 100

	print percentage

if __name__ == "__main__":
    main()






