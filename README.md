#视频流行度预测系统

###运行环境
- python3
- Tensorflow r1.0

###运行方法
1. 首先运行DataGenerator.py文件生成训练数据和测试数据，这两个数据集以json格式保存在项目的data文件夹下面
2. 然后运行Test.py方法，这个方法会根据KMeans聚类结果训练对应的神经网络，然后应用测试集数据进行测试