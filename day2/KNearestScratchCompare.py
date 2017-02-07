from math import sqrt
import numpy as np 
from matplotlib import style
import matplotlib.pyplot as plt 
import warnings
from collections import Counter
import pandas as pd 
import random
style.use('fivethirtyeight')

#dataset = {'k' : [[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}

# Can see all the points using this code
new_features = [5,7]
# for i in dataset:
# 	for ii in dataset[i]:
# 		plt.scatter(ii[0],ii[1], s = 100, color = i)

# plt.scatter(new_features[0], new_features[1], s = 100, color ='g')
# plt.show()


def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups')

	distances = []

#will have to explain about the formulae used here. its part of numpy and quicker than conventional methods
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	print(Counter(votes).most_common(1))

	# this gives the first element, its value and its count
	vote_result = Counter(votes).most_common(1)[0][0]

	#can also brief them about confidence and the differnce from accuracy

	confidence = Counter(votes).most_common(1)[0][1] / k
	return vote_result, confidence


df = pd.read_csv('breast-cancer-wisconsin.txt')

df.replace('?', -99999, inplace =True)
df.drop(['id'], 1, inplace=True)

#need conversion as some variables are strings or in quotes
full_data = df.astype(float).values.tolist()

#print(full_data[:10])

random.shuffle(full_data)

test_size = 0.2

train_set = {2:[], 4:[]}

test_set = {2:[], 4:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

#creating train and test datasets
#the respective values are apppended to the required dataset
for i in train_data:
	train_set[i[-1]].append(i[:-1])


for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		#can print confidence if required, not doing anything as such for now
		vote, confidence = k_nearest_neighbors(train_set, data, k=5)
		if group == vote:
			correct += 1
		total += 1

print('Accuracy:', correct/total)