import math
import matplotlib.pyplot as plt
import numpy as np

from DataPipeline import DataPipeline
from Network import Network
from Constant import *
import Cluster

def test():
	dataDict,model = Cluster.cluster()
	#根据聚类模型，分割验证集
	valiDict = {}
	for key in range(len(dataDict)):
		valiDict[key] = []
	valiPipe = DataPipeline('data/ValiData.json')
	for item in valiPipe.trainList():
		clusterType = model.predict([item[FEATURE_START_INDEX:FEATURE_START_INDEX + NUM_FEATURES]])[0]
		valiDict[clusterType].append(item)
	#根据聚类数据训练网络
	netDict = {}
	for key in dataDict:
		netDict[key] = Network(key + 1, valiDict[key])
		print("======network %d=====" % (key + 1))
		netDict[key].train(DataPipeline(None, isFile=False, list=dataDict[key]))

	centers = model.cluster_centers_
	drawCenters(centers)
	#测试
	testPipe = DataPipeline('data/TestData.json')
	testNum = testPipe.size()
	testData = testPipe.trainList()
	result_list = []
	acc_list = []
	for i in range(testNum):
		clusterType = model.predict([testData[i][FEATURE_START_INDEX:FEATURE_START_INDEX + NUM_FEATURES]])[0]
		print("cluster:%d" % (clusterType + 1))
		predict_result = netDict[clusterType].predict([testData[i][FEATURE_START_INDEX:FEATURE_START_INDEX + NUM_FEATURES]])[0][0]
		real_result = sum(testData[i][FEATURE_START_INDEX:FEATURE_START_INDEX + TOTAL_FEATURES_NUM])
		
		acc = error(real_result, predict_result)
		result_list.append((real_result, predict_result))
		acc_list.append(acc) 

	for item in result_list:
		print("%.2f=====%.2f" % (item[0], item[1]))
	rmse = rmsd(result_list)
	accuracy = sum(acc_list) / len(acc_list)
	print("test rmsd = %.2f,test accuracy = %.2f%%" % (rmse, 100. * accuracy))

def drawCenters(centers):
	x = np.array(list(range(1, 1 + NUM_FEATURES)))
	for index in range(centers.shape[0]):
		plt.figure(index)
		plt.clf()
		plt.plot(x, centers[index, :])
		figureName = "graph/center" + str(index + 1) + ".png"
		plt.savefig(figureName)
	
def error(realY, predictY):
	acc = 1 - abs(realY - predictY) / realY
	return acc

def rmsd(result_list):
	total = 0
	for item in result_list:
		total += math.pow(item[0] - item[1], 2)
	return math.sqrt(total / len(result_list))

def main():
	test()

if __name__ == '__main__':
	main()
