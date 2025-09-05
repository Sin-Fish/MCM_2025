import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.util.data_manager import Data
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

    def plot_clusters(self, X, x_col_name=None, y_col_name=None):
        '''绘制聚类结果
        Args:
            X: 特征矩阵 (n_samples, n_features)
            x_col_name: x轴特征名称
            y_col_name: y轴特征名称
        '''
        if self.labels_ is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 设置中文字体支持
        self._set_chinese_font()
        
        # 获取数据
        x_data = X.iloc[:, 0] if x_col_name is None else X[x_col_name]
        y_data = X.iloc[:, 1] if y_col_name is None else X[y_col_name]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 为每个簇绘制不同颜色的点
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
        for i in range(self.n_clusters):
            cluster_points = self.labels_ == i
            plt.scatter(x_data[cluster_points], y_data[cluster_points], 
                       c=[colors[i]], label=f'簇 {i}', alpha=0.7, s=50)
        
        # 绘制聚类中心
        if self.cluster_centers_ is not None:
            center_x = self.cluster_centers_[:, 0] if x_col_name is None else self.cluster_centers_[:, X.columns.get_loc(x_col_name)]
            center_y = self.cluster_centers_[:, 1] if y_col_name is None else self.cluster_centers_[:, X.columns.get_loc(y_col_name)]
            plt.scatter(center_x, center_y, c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        
        # 添加标签和标题
        plt.xlabel(x_col_name if x_col_name else X.columns[0])
        plt.ylabel(y_col_name if y_col_name else X.columns[1])
        plt.title('K-means聚类结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def _set_chinese_font(self):
        '''设置中文字体支持'''
        # 定义可能的中文字体
        font_names = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'FangSong']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 尝试设置中文字体
        for font_name in font_names:
            if font_name in available_fonts:
                plt.rcParams['font.sans-serif'] = [font_name]
                break
        else:
            # 如果常用字体不可用，尝试使用其他中文字体
            chinese_fonts = [f for f in available_fonts if any(chinese_char in f for chinese_char in ['Sim', 'Kai', 'Fang', 'Microsoft', 'ST'])]
            if chinese_fonts:
                plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
        
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    data = Data.data
    #feature = ['检测孕周', '孕妇BMI']
    feature = ['Y染色体浓度', '孕妇BMI']
    # 选择用于聚类的特征
    X = data[feature].dropna()
    
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
    
    # 绘制聚类结果
    kmeans.plot_clusters(X, feature[0], feature[1])