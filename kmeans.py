
#encoding=utf8

import random
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X): #X为初始三个点
    one_sample = one_sample.reshape(1, -1) #reshape 为 一行
    #print "one_sample",one_sample
    #print 'X_org',X
    X = X.reshape(X.shape[0], -1)
    #print 'X',X
    print 'tail',np.tile(one_sample, (X.shape[0], 1)) #复制函数 tail => [[1.48342126  0.6382319   1.06795023],
                                                                         #[1.48342126  0.6382319   1.06795023],
    #矩阵做差后平方，并根据行求和,((x1-y1)^2+(x2-y2)^2)^1/2              #[1.48342126  0.6382319   1.06795023]]
    print 'power',np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1) #distances => [3.95019696  11.86337976   0.90384074]
    return distances



class Kmeans():
    """Kmeans聚类算法.

    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数. 
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 
        则说明算法已经收敛
    """
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)#取最小值对应的索引号
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        #print 'np.shape(X)',np.shape(X)
        n_samples = np.shape(X)[0] #np.shape(X)表示获取np数组行列
        print 'n_samples',n_samples
        clusters = [[] for _ in range(self.k)] #clusters => [[],[],[]]
        for sample_i, sample in enumerate(X):
            #遍历计算X中每个list 与 中心点距离并返回最近中心点索引
            centroid_i = self._closest_centroid(sample, centroids)#返回与sample最近的索引
            #print 'centroid_i',centroid_i
            clusters[centroid_i].append(sample_i)
            #print 'clusters',clusters
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1] # =>3
        centroids = np.zeros((self.k, n_features)) #=>3*3
        print 'clusters', clusters
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            print 'centroid', centroid
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])#行数[ 0.  0.  0.  0.  ...]
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)
        print 'initial k',centroids
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X) #输入随机选取的中心点以及X
            former_centroids = centroids #初始的k

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X) #=>[ 0.42649428  0.48287368  0.36232152] 新中心点
            
            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon: #diff.any() 查看diff 中 三个特征变化是否都小于varepsilon
                break

        return self.get_cluster_labels(clusters, X)


def main():
    # Load the dataset
    X, y = datasets.make_blobs(n_samples=30, 
                               n_features=3, 
                               centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], 
                               cluster_std=[0.2, 0.1, 0.2, 0.2], 
                               random_state =9)
                               
    print 'X',X
    print 'y',y

    # 用Kmeans算法进行聚类
    clf = Kmeans(k=3)
    y_pred = clf.predict(X)

    print 'y_pred',y_pred


if __name__ == "__main__":
    main()
