#-*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from DataPipeline import DataPipeline
from Constant import *
from DataGenerator import saveFile

def error(realY, predictY):
	acc = 1 - abs(realY - predictY) / realY
	return acc
		
class Network(object):
	def __init__(self, index, vali):
		super(Network, self).__init__()
		#input and output placeholder
		X = tf.placeholder(tf.float64, [None, NUM_FEATURES])
		Y = tf.placeholder(tf.float64, [None, NUM_OUTPUT])
	
		#weight and bias
		weight1 = self.initWeight((NUM_FEATURES, NUM_HIDDEN))
		bias1 = self.initBias([NUM_HIDDEN])
		weight2 = self.initWeight((NUM_HIDDEN,NUM_HIDDEN))
		bias2 = self.initBias([NUM_HIDDEN])
		weight3 = self.initWeight((NUM_HIDDEN,NUM_HIDDEN))
		bias3 = self.initBias([NUM_HIDDEN])
		weight4 = self.initWeight((NUM_HIDDEN,NUM_OUTPUT))
		bias4 = self.initBias([NUM_OUTPUT])
	
	  	#relation
		hidden1 = tf.matmul(X,weight1) + bias1
		hidden2 = tf.matmul(hidden1,weight2) + bias2
		hidden3 = tf.matmul(hidden2,weight3) + bias3
		yLogits = tf.matmul(hidden3,weight4) + bias4
	
	  	#back propagation
		cost    = tf.reduce_sum(tf.pow(yLogits-Y, 2))
		updates = tf.train.GradientDescentOptimizer(0.0000000001).minimize(cost)
	
		#run LGD
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		self.session = sess
		self.inputX = X
		self.inputY = Y
		self.update = updates
		self.logitsY = yLogits
		self.accuracyList = []
		self.index = index
		self.valiData = vali

	def __del__(self):
		self.session.close()

	def initWeight(self, shape):
		weights = tf.random_normal(shape, stddev=0.1, dtype=tf.float64)
		return tf.Variable(weights,dtype=tf.float64)

	def initBias(self, shape):
		bias = tf.zeros(shape, dtype=tf.float64)
		return tf.Variable(bias, dtype=tf.float64)
	
	def train(self, trainData):
		print('=====start training=====')
		for epoch in range(NUM_EPOCH):
			for i in range(EPOCH_SAMPLE):
				line = trainData.nextRecord()
				self.session.run(self.update,feed_dict={self.inputX:[line[FEATURE_START_INDEX:FEATURE_START_INDEX + NUM_FEATURES]], \
					self.inputY:[[sum(line[FEATURE_START_INDEX:FEATURE_START_INDEX + TOTAL_FEATURES_NUM])]]})
			record = trainData.nextRecord()
			print("the epoch %d" % (epoch + 1))
			self.validate()
		print('=====end training=====')
		plt.figure(self.index)
		x = list(range(1,len(self.accuracyList) + 1))
		plt.plot(np.array(x), np.array(self.accuracyList))
		plt.xlabel(u'轮数',fontproperties=HEITI)
		plt.ylabel(u'精确度',fontproperties=HEITI)
		figureName = "graph/networkAcc" + str(self.index) + ".png"
		plt.savefig(figureName)
		fileName = "graph/networkAcc" + str(self.index) + ".json"
		saveFile(fileName,self.accuracyList)

	def validate(self):
		if len(self.valiData) == 0:
			valiPipe = DataPipeline('data/ValiData.json')
			self.valiData = valiPipe.trainList()
		valiList = []
		for record in self.valiData:
			acc = error(sum(record[FEATURE_START_INDEX:FEATURE_START_INDEX + TOTAL_FEATURES_NUM]), \
				self.predict([record[FEATURE_START_INDEX:FEATURE_START_INDEX + NUM_FEATURES]])[0][0])
			valiList.append(acc)
		accuracy = sum(valiList) / len(valiList)
		self.accuracyList.append(accuracy)
		print("the accuracy %.2f%%" % (100. * accuracy))
		
			
	def predict(self, inp):
		predict = self.session.run(self.logitsY, feed_dict={self.inputX: inp})
		return predict


