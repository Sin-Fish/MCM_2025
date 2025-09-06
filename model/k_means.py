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

    def remap_labels_by_first_feature(self):
        '''重新映射聚类标签，使标签按聚类中心第一个特征值递增排序
        '''
        if self.cluster_centers_ is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 获取聚类中心
        centers = self.get_cluster_centers()
        
        # 根据第一个特征对标签进行排序
        sorted_indices = np.argsort(centers[:, 0])
        
        # 创建标签映射
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        
        # 重新映射训练数据的标签
        remapped_labels = np.array([label_mapping[label] for label in self.get_labels()])
        self.labels_ = remapped_labels
        
        # 重新映射聚类中心顺序
        remapped_centers = centers[sorted_indices]
        self.cluster_centers_ = remapped_centers
        
        # 更新模型的预测方法，使其使用新的标签映射
        original_predict = self.model.predict
        def remapped_predict(X):
            original_labels = original_predict(X)
            return np.array([label_mapping[label] for label in original_labels])
        
        self.model.predict = remapped_predict

    def print_cluster_boundaries(self, X, x_col_name=None, y_col_name=None):
        '''输出每类的边界数据，帮助确定区间
        Args:
            X: 特征矩阵 (n_samples, n_features)
            x_col_name: x轴特征名称
            y_col_name: y轴特征名称
        '''
        if self.labels_ is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 获取特征名称
        x_col = x_col_name if x_col_name else X.columns[0]
        y_col = y_col_name if y_col_name and y_col_name in X.columns else (X.columns[1] if X.shape[1] > 1 else None)
        
        print("\n=== 各簇边界信息 ===")
        for i in range(self.n_clusters):
            cluster_points = self.labels_ == i
            cluster_data = X[cluster_points]
            
            print(f"\n簇 {i} 的边界信息:")
            
            # 输出Y轴（第一特征）的区间情况
            x_data = cluster_data[x_col]
            print(f"  {x_col} 范围: [{x_data.min():.2f}, {x_data.max():.2f}]")
            print(f"  {x_col} 中位数: {x_data.median():.2f}")
            
            # 如果有第二特征，输出X轴（第二特征）的区间情况
            if y_col:
                y_data = cluster_data[y_col]
                print(f"  {y_col} 范围: [{y_data.min():.2f}, {y_data.max():.2f}]")
                print(f"  {y_col} 中位数: {y_data.median():.2f}")
                
            print(f"  样本数量: {len(cluster_data)}")

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
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 如果只有一列特征，生成虚拟y轴数据
        if X.shape[1] == 1 or y_col_name is None:
            y_data = np.zeros(len(x_data))  # 创建零值作为y轴
            # 为每个簇绘制不同颜色的点
            colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
            for i in range(self.n_clusters):
                cluster_points = self.labels_ == i
                plt.scatter(x_data[cluster_points], y_data[cluster_points], 
                           c=[colors[i]], label=f'簇 {i}', alpha=0.7, s=50)
            
            # 绘制聚类中心
            if self.cluster_centers_ is not None:
                center_x = self.cluster_centers_[:, 0]
                center_y = np.zeros(len(center_x))
                plt.scatter(center_x, center_y, c='red', marker='x', s=200, linewidths=3, label='聚类中心')
            
            plt.xlabel(x_col_name if x_col_name else X.columns[0])
            plt.ylabel('虚拟轴')
        else:
            # 如果有两列特征，正常绘制二维散点图
            y_data = X[y_col_name]
            
            # 为每个簇绘制不同颜色的点
            colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
            for i in range(self.n_clusters):
                cluster_points = self.labels_ == i
                plt.scatter(x_data[cluster_points], y_data[cluster_points], 
                           c=[colors[i]], label=f'簇 {i}', alpha=0.7, s=50)
            
            # 绘制聚类中心
            if self.cluster_centers_ is not None:
                center_x = self.cluster_centers_[:, X.columns.get_loc(x_col_name)]
                center_y = self.cluster_centers_[:, X.columns.get_loc(y_col_name)]
                plt.scatter(center_x, center_y, c='red', marker='x', s=200, linewidths=3, label='聚类中心')
            
            plt.xlabel(x_col_name if x_col_name else X.columns[0])
            plt.ylabel(y_col_name)
        
        # 添加标题和图例
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

def find_optimal_clusters(X, max_clusters=10, random_state=42):
    '''使用肘部法则确定最佳聚类数
    Args:
        X: 特征矩阵 (n_samples, n_features)
        max_clusters: 最大聚类数
        random_state: 随机种子
    Returns:
        inertia_values: 每个聚类数对应的惯性值列表
    '''
    inertia_values = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
    
    return list(K_range), inertia_values

def plot_elbow_method(X, max_clusters=10, random_state=42):
    '''绘制肘部法则图
    Args:
        X: 特征矩阵 (n_samples, n_features)
        max_clusters: 最大聚类数
        random_state: 随机种子
    '''
    # 设置中文字体支持
    plt.rcParams['axes.unicode_minus'] = False
    font_names = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'FangSong']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in font_names:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    
    K_range, inertia_values = find_optimal_clusters(X, max_clusters, random_state)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia_values, 'bo-')
    plt.xlabel('聚类数 (k)')
    plt.ylabel('簇内平方和 (惯性值)')
    plt.title('肘部法则确定最佳聚类数')
    plt.grid(True, alpha=0.3)
    
    # 标注每个点的值
    for i, (k, inertia) in enumerate(zip(K_range, inertia_values)):
        plt.annotate(f'{inertia:.0f}', (k, inertia), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.show()
    
    return K_range, inertia_values

if __name__ == "__main__":
    data = Data.data
    # 选择用于聚类的特征
    #feature = ['检测孕周', '孕妇BMI']
    feature = ['孕妇BMI',"Y染色体浓度"]
    
    X = data[feature].dropna()
    
    # 创建并训练K-means模型
    kmeans = KMeansCluster(n_clusters=3)
    kmeans.train(X)
    
    plot_elbow_method(X, max_clusters=10)
    # 重新映射标签，使其按照第一个特征值排序
    kmeans.remap_labels_by_first_feature()
    
    # 输出结果
    print("聚类中心:")
    print(kmeans.get_cluster_centers())
    print("惯性 (簇内平方和):", kmeans.get_inertia())
    print("前10个样本的聚类标签:", kmeans.get_labels()[:10])
    
    # 输出各类的边界数据
    if len(feature) == 2:
        kmeans.print_cluster_boundaries(X, feature[0], feature[1])
    elif len(feature) == 1:
        kmeans.print_cluster_boundaries(X, feature[0])
    
    # 预测新数据点的聚类
    sample_data = X[:5]
    predictions = kmeans.predict(sample_data)
    print("\n示例数据的聚类预测:", predictions)
    
    # 绘制聚类结果
    if len(feature) == 2:
        kmeans.plot_clusters(X, feature[0], feature[1])
    elif len(feature) == 1:
        kmeans.plot_clusters(X, feature[0])