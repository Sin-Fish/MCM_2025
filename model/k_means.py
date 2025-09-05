import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.util.data_manager import Data
from sklearn.cluster import KMeans
import numpy as np

class KMeansCluster:
    def __init__(self, n_clusters=3, random_state=42):
        '''初始化K-means聚类模型
        Args:
            n_clusters: 聚类数量
            random_state: 随机种子
        '''
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = None

    def train(self, X):
        '''训练K-means聚类模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
        '''
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_

    def predict(self, X):
        '''预测样本所属的簇
        Args:
            X: 特征矩阵 (n_samples, n_features)
        Returns:
            预测的簇标签
        '''
        return self.model.predict(X)

    def get_cluster_centers(self):
        '''获取聚类中心
        Returns:
            聚类中心坐标
        '''
        return self.cluster_centers_

    def get_labels(self):
        '''获取训练数据的聚类标签
        Returns:
            训练数据的聚类标签
        '''
        return self.labels_

    def get_inertia(self):
        '''获取聚类的惯性（簇内平方和）
        Returns:
            惯性值
        '''
        return self.model.inertia_

if __name__ == "__main__":
    data = Data.data
    # 选择用于聚类的特征
    X = data[['检测孕周', '孕妇BMI']].dropna()
    
    # 创建并训练K-means模型
    kmeans = KMeansCluster(n_clusters=3)
    kmeans.train(X)
    
    # 输出结果
    print("聚类中心:")
    print(kmeans.get_cluster_centers())
    print("惯性 (簇内平方和):", kmeans.get_inertia())
    print("前10个样本的聚类标签:", kmeans.get_labels()[:10])
    
    # 预测新数据点的聚类
    sample_data = X[:5]
    predictions = kmeans.predict(sample_data)
    print("示例数据的聚类预测:", predictions)