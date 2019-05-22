#!//usr/bin/python3.6

import numpy as np
from sklearn import preprocessing

splitNum = 405
numBins = 2
bins = np.linspace(0, 1, numBins + 1)
idToData = {}
scaler = None

def loadTrainLabels(path):
	pairs = np.loadtxt(path, delimiter=',', skiprows=1, dtype=int)
	# print(pairs) #############

	mp = {}
	for p in pairs:
		mp[p[0]] = p[1]

	return pairs[:, 0], pairs, mp

def sortData():
	# print(idLabelPairs[idLabelPairs[:, 1] == 7])

	idsOfUser = np.zeros((21, 450), dtype=int)
	for u in range(1, 21):
		idsOfUser[u, :] = idLabelPairs[ idLabelPairs[:, 1] == u, 0]

	return idsOfUser

def splitData():
	trainArrays = []
	validationArrays = []

	for u in range(1, 21):
		trainArrays.append( idsOfUser[u, 0:splitNum] )
		validationArrays.append( idsOfUser[u, splitNum:] )

	trainId = np.concatenate(tuple(trainArrays), axis=0)
	validationId = np.concatenate(tuple(validationArrays), axis=0)
	return trainId, validationId

def loadData(path, ids):
	count = 0
	for id in ids:
		count += 1
		idToData[id] = np.loadtxt(path + str(id) + ".csv", delimiter=',', skiprows=0, dtype=np.float64)

		##################################
		if (count % 100 == 0):
			print("loaded", count, "data")
			# print(idToData[id]) 
			print("=" * 20, "\n")

def fitScaler():
	# xArrays = []
	# yArrays = []
	# zArrays = []
	# for id in trainId:
	# 	array = idToData[id]
	# 	xArrays.append(array[:, 0])
	# 	yArrays.append(array[:, 1])
	# 	zArrays.append(array[:, 2])

	# xValues = np.concatenate(tuple(xArrays), axis=0)
	# yValues = np.concatenate(tuple(yArrays), axis=0)
	# zValues = np.concatenate(tuple(zArrays), axis=0)

	# print(xValues) ###########


	# xScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	# xScaler.fit(xValues.reshape())

	# yScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	# yScaler.fit(yValues)

	# zScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	# zScaler.fit(zValues)

	# for array in idToData.values():
	# 	xScaler.fit(array[:, 0])
	# 	yScaler.fit(array[:, 1])
	# 	zScaler.fit(array[:, 2])


	arrays = []
	for id in trainId:
		arrays.append(idToData[id])

	values = np.concatenate(tuple(arrays), axis=0)

	global scaler
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	scaler.fit(values)


	print("Fitted the scaler")


def normalizeMinMax(ids):
	for id in ids:
		idToData[id] = scaler.transform(idToData[id])

	print("Normalized data")

def interpolateData(ids):
	for id in ids:
		array = idToData[id]
		newArray = np.zeros((150, 3))

		x = np.arange(1, 151, 1)	
		xp = np.linspace(1, 150, array.shape[0])
		for k in range(0, 3):
			yp = array[:, k]
			newArray[:, k] = np.interp(x, xp, yp)

		idToData[id] = newArray

	print("interpolized data")

def pointToBin(point):
	index = np.digitize(point, bins) - 1

	ans = 0
	for k in range(point.size):
		ans += index[k] * numBins ** (point.size - k - 1)

	return ans


def computeProbabilityMatrix():
	probMatrix = np.zeros((150, numBins ** 3, 21))

	for userIndex in range(1, 21):
		resultBins = np.zeros(idsOfUser[userIndex].size, dtype=int)
		for featureIndex in range(150):

			for i in range(idsOfUser[userIndex].size):
				id = idsOfUser[userIndex][i]
				data = idToData[id]
				resultBins[i] = pointToBin(data[featureIndex, :])

			for binIndex in range(numBins ** 3):
				probMatrix[featureIndex][binIndex][userIndex] = np.sum(resultBins == binIndex) / (resultBins.size)

	print("computed probability matrix")
	return probMatrix

def computeLabels(ids):
	result = np.zeros((ids.shape[0], 21))

	for i in range(ids.shape[0]):
		id = ids[i]
		data = idToData[id]

		for user in range(1, 21):

			prob = 0
			for featureIndex in range(150):
				point = data[featureIndex, :]

				print("point,binOfPoint = ", point, ", ", pointToBin(point))

				prob += np.log(probMatrix[featureIndex, pointToBin(point), user] + 1e-10)

			result[i, user] = prob

		print("computed result for", id, "; i = ", i)
		print(" class is: ", np.argmax(result[i, 1:]) + 1)

	return np.argmax(result[:, 1:], axis = 1) + 1

def getAccuracy(actual, predicted):
	return np.sum(actual == predicted).sum() / actual.shape[0] * 100



def doValidation():
	predictedLabels = computeLabels(validationId)
	print("predicted label - id")
	print(validationId)
	print(predictedLabels)


	actualLabels = np.zeros(validationId.shape[0])
	for i in range(validationId.shape[0]):
		actualLabels[i] = idToLabel[validationId[i]]

	accuracy = getAccuracy(actualLabels, predictedLabels)
	print("accuracy =", accuracy)

def getLabelsFor(ids):
	predictedLabels = computeLabels(ids)	
	return predictedLabels






allTrainId, idLabelPairs, idToLabel = loadTrainLabels("../../data/train_labels.csv")
# print(allTrainId) #################
# print(idToLabel)



# test PointToBin
# print(bins)
# test = np.array([[0.3, 0.6]] * 3)
# print(test)
# for i in range(test.shape[1]):
# 	for j in range(test.shape[1]):
# 		for k in range(test.shape[1]):
# 			point = np.array((test[0, i], test[1, j], test[2, k]))
# 			print(point, pointToBin(point))




idsOfUser = sortData()
# print(idsOfUser)
# print('=' * 20)

trainId, validationId = splitData()
###########################################################################
# print(trainId.size)
# print(validationId.size)
# print("=" * 20)
# print(trainId)
# print(validationId)


loadData("../../data/train/", allTrainId)
# print(idToData) ##################
# print(list(idToData.values()))


fitScaler()
normalizeMinMax(np.concatenate((trainId, validationId)))
interpolateData(np.concatenate((trainId, validationId)))


# print("normalizedData of 23999:") ###########
# print(idToData[23999])



probMatrix = computeProbabilityMatrix()


# doValidation()



testId = np.loadtxt("../test_files.txt", dtype=int)
testId = testId[0 : (int)(0.2 * testId.size)] ########################################################## WATCH ME ######################
# print("testId ndarray", testId, "; size = ", testId.shape)

loadData("../../data/test/", testId)
normalizeMinMax(testId)
interpolateData(testId)


# i = 948
# print("i = ", i, "testId[i] = ", testId[i])
# print(idToData[testId[i]])

testId = np.array(list(map(lambda x: testId[x], [947, 948, 949, 950]))) ######################################################## WATCH ME #################

print(testId) ######
print(testId.shape)

testLabels = getLabelsFor(testId)
print(np.concatenate((testId, testLabels), axis = 1))
