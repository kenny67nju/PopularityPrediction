from dtw import dtw
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances

def dtwDistance(x,y):
	#使用dtw库计算dtw距离，使用欧几里得距离作为cost衡量函数
	shapedX = x.reshape(-1, 1)
	shapedY = y.reshape(-1, 1)
	dist, _, _, _ = dtw(shapedX, shapedY, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
	return dist


#come from github:https://github.com/letiantian/kmedoids
def kMedoids(D, k, tmax=100):
	# determine dimensions of distance matrix D
	m, n = D.shape

	if k > n:
		raise Exception('too many medoids')
	# randomly initialize an array of k medoid indices
	M = np.arange(n)
	np.random.shuffle(M)
	M = np.sort(M[:k])

	# create a copy of the array of medoid indices
	Mnew = np.copy(M)

	# initialize a dictionary to represent clusters
	C = {}
	for t in range(tmax):
		# determine clusters, i. e. arrays of data indices
		J = np.argmin(D[:,M], axis=1)
		for kappa in range(k):
			C[kappa] = np.where(J==kappa)[0]
		# update cluster medoids
		for kappa in range(k):
			J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
			j = np.argmin(J)
			Mnew[kappa] = C[kappa][j]
		np.sort(Mnew)
		# check for convergence
		if np.array_equal(M, Mnew):
			break
		M = np.copy(Mnew)
	else:
		# final update of cluster memberships
		J = np.argmin(D[:,M], axis=1)
		for kappa in range(k):
			C[kappa] = np.where(J==kappa)[0]

	# return results
	return M, C

class KMedoids(object):
	"""k中心点聚类法的抽象，使用DTW作为距离衡量方法"""
	def __init__(self, n_cluster):
		super(KMedoids, self).__init__()
		self.num = n_cluster
		self.medoids = {}
		self.label = []

	def fit(self, data):
		size = data.shape[0]
		# D = pairwise_distances(data, metric='euclidean')
		D = pairwise_distances(data, metric=dtwDistance)
		print("=====距离计算完毕=====")
		M, C = kMedoids(D, self.num)
		print("=====KMedoids聚类完毕=====")
		for point_idx in M:
			for label in C:
				if point_idx in C[label]:
					self.medoids[label] = data[point_idx]

		reverseDict = {}
		for label in C:
			for point_idx in C[label]:
				reverseDict[point_idx] = label
		for index in range(size):
			self.label.append(reverseDict[index])
		return self

	def labels(self):
		return self.label

	def predict(self, features):
		result = []
		for feature in features:
			target = 0
			distance = float("inf")
			for label in self.medoids:
				newDistance = dtwDistance(self.medoids[label],np.array(feature))
				if newDistance < distance:
					distance = newDistance
					target = label
			result.append([target])
		return result


#测试方法
# if __name__ == '__main__':
# 	estimator = KMedoids(2)
# 	data = np.array([[1,1,10], [2,2,2], [10,10,100],[100,100,100]])
# 	estimator = estimator.fit(data)
# 	print(estimator.labels())
# 	print(estimator.predict([[200,200,200]]))


		
		