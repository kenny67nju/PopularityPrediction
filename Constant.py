import matplotlib

NUM_FEATURES = 3 * 24 #训练使用的特征的数量
NUM_HIDDEN = 100 #隐藏层网络的节点数
NUM_OUTPUT = 1 #输出层节点数目
TOTAL_FEATURES_NUM = 7 * 24 #总特征的数量
NUM_CLUSTERS = 4 #聚类的个数
NUM_EPOCH = 3000 #训练的轮数
EPOCH_SAMPLE = 100 #每轮训练的样本个数
FEATURE_START_INDEX = 2 #特征开始的index，特征前存放名称之类的信息
CLUSTER_TYPE = 'KMEANS'#支持'KMEANS','KMEDOIDS','SPECTRAL'
HEITI = matplotlib.font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')#中文字体