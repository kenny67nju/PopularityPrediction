#-*-coding:utf-8 -*-
import xlrd
import json
import random



#从Excel读取数据表
def openExcel(file = 'data/PopularityData.xlsx'):
	try:
		data = xlrd.open_workbook(file)
		return data
	except Exception as e:
		print(str(e))

#将数据分割成训练集，验证集，测试集
def splitData():
	data = openExcel()
	table = data.sheets()[0]
	print("=====读入数据表=====")
	rowsNum = (table.nrows - 1)
	testDataNum = rowsNum // 5
	trainDataNum = rowsNum - testDataNum
	valiDataNum = trainDataNum // 100 	
	allData = []
	for i in range(rowsNum):
		row = table.row(i+1)
		data = [i+1,row[0].value]
		data.extend([0.0 if row[index].value == '' else row[index].value for index in range(21,21 + 168)])
		# if(checkNull(data)):
		# 	continue
		# if(0.0 in data):
		# 	continue
		allData.append(data)
		print(data)
	print('=====shuffle starts=====')
	random.shuffle(allData)
	print('=====shuffle ends=====')
	trainData = []
	testData = []
	valiData = []
	for i in range(len(allData)):
		data = allData[i]
		if int(data[0]) <= valiDataNum:
			valiData.append(data)
		elif int(data[0]) > valiDataNum and int(data[0]) < trainDataNum:
			trainData.append(data)
		else:
			testData.append(data)
	saveFile('data/ValiData.json',valiData)
	saveFile('data/TrainData.json',trainData)
	saveFile('data/TestData.json',testData)

def preProcess(data):
	total = 0
	count = 0
	for num in data[FEATURE_START_INDEX:]:
		if num != 0.0:
			total += num
			count += 1
	mean = total / count
	for i in range(1,len(data)):
		if data[i] == 0.0:
			data[i] = round(mean,0)

def checkNull(data):
	isNull = True
	for num in data[FEATURE_START_INDEX:]:
		if num != 0.0:
			return False
	return isNull

def saveFile(file,data):
	with open(file, 'w') as f:
		json.dump(data,f)

def main():
	splitData()

if __name__ == '__main__':
	main()