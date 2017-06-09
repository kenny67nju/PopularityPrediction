#-*-coding:utf-8 -*-
import json

class DataPipeline(object):
	"""the pipeline of imput data"""
	def __init__(self, file, isFile=True, list=None):
		super(DataPipeline, self).__init__()
		if isFile:
			with open(file,'r') as f:
				data = json.load(f)
				self.resource = data
				self.len = len(data)
				self.offset = 0
		else:
			self.resource = list
			self.len = len(list)
			self.offset = 0
	
	def size(self):
		return self.len

	def reset(self):
		self.offset = 0

	def trainList(self):
		return self.resource

	def nextRecord(self):
		record = self.resource[self.offset]
		self.offset += 1
		self.offset = self.offset % self.len
		return record

	def nextBatch(self,batchSize=200):
		if batchSize > self.len:
			return

		result = []
		for i in range(self.offset,self.offset + batchSize):
			result.append(self.resource[i % self.len])

		self.offset += batchSize
		self.offset = self.offset % self.len
		return result


# pipeline = DataPipeline('data/TestData.json')
# for item in pipeline.nextBatch(20):
# 	print(item)
# for i in range(10000):
# 	print(pipeline.nextRecord()[0])
