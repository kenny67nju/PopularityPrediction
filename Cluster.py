#-*-coding:utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np

from Constant import *
from DataPipeline import DataPipeline
from KMedoids import KMedoids

def calOutlines(trainPipe):
	outlines = []
	for _ in range(trainPipe.size()):
		line = trainPipe.nextRecord()
		outlines.append(line[0:FEATURE_START_INDEX + NUM_FEATURES])
	return np.array(outlines)


def cluster():
	trainPipe = DataPipeline('data/TrainData.json')
	print("=====训练数据读入完毕=====")
	outlines = calOutlines(trainPipe)
	if CLUSTER_TYPE == 'KMEDOIDS':
		kmedoids = KMedoids(NUM_CLUSTERS)
		kmedoids = kmedoids.fit(outlines[:,FEATURE_START_INDEX:])
		clusterLabel = kmedoids.labels()
		resource = trainPipe.trainList()
		dataDict = {}
		for i in range(NUM_CLUSTERS):
			dataDict[i] = []
		for i in range(len(clusterLabel)):
			dataDict[clusterLabel[i]].append(resource[i])
		return dataDict,kmedoids
	elif CLUSTER_TYPE == 'KMEANS':
		kmeans = KMeans(n_clusters = NUM_CLUSTERS).fit(outlines[:,FEATURE_START_INDEX:])
		clusterLabel = kmeans.labels_
		resource = trainPipe.trainList()
		dataDict = {}
		for i in range(NUM_CLUSTERS):
			dataDict[i] = []
		for i in range(len(clusterLabel)):
			dataDict[clusterLabel[i]].append(resource[i])
		return dataDict,kmeans
	else:
		raise Exception('Cluster Type Error')

# if __name__ == '__main__':
# 	dataDict,model = cluster()
# 	print(dataDict)