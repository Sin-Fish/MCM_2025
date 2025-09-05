import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 修复路径问题 - 使用相对路径添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.util.data_manager import Data
from model.k_means import KMeansCluster
from data.util.draw import draw_col_seaborn
from model.logistic_regression import LogisticRegressionModel

def plot_cluster_data(cluster_id, data, title):
    """绘制簇数据的散点图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data['检测孕周'], data['Y染色体浓度'], alpha=0.7)
    plt.xlabel('检测孕周')
    plt.ylabel('Y染色体浓度')
    plt.title(f'{title} - 簇 {cluster_id}')
    plt.grid(True, alpha=0.3)
    plt.show()

def days_to_weeks_days(days):
    """将天数转换为周数和天数的格式"""
    weeks = int(days // 7)
    remaining_days = int(days % 7)
    return weeks, remaining_days

def format_gestational_age(days):
    """格式化孕周显示"""
    weeks, days = days_to_weeks_days(days)
    if days == 0:
        return f"{weeks}周"
    else:
        return f"{weeks}周+{days}天"

if __name__ == "__main__":
    data = Data.data
   #统计每个孕妇达标孕周，
    #feature = ['检测孕周', '孕妇BMI']
    feature = ['孕妇BMI',"Y染色体浓度"]
    
    X = data[feature].dropna()
    
 
    kmeans = KMeansCluster(n_clusters=4)
    kmeans.train(X)
    
   
    print("聚类中心:")
    print(kmeans.get_cluster_centers())
    print("惯性 (簇内平方和):", kmeans.get_inertia())
    print("前10个样本的聚类标签:", kmeans.get_labels()[:10])
    
    
    if len(feature) == 2:
        kmeans.print_cluster_boundaries(X, feature[0], feature[1])
    elif len(feature) == 1:
        kmeans.print_cluster_boundaries(X, feature[0])
    
    
    sample_data = X[:5]
    predictions = kmeans.predict(sample_data)
    print("\n示例数据的聚类预测:", predictions)
    
    
    if len(feature) == 2:
        kmeans.plot_clusters(X, feature[0], feature[1])
    elif len(feature) == 1:
        kmeans.plot_clusters(X, feature[0])

    # 导入first_y.csv,并根据聚类结果分离不同类别的数据
    first_y_file_path = os.path.join(project_root, "data", "first_y.csv")
    
    # 尝试不同的编码方式读取CSV文件
    try:
        first_y_data = pd.read_csv(first_y_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            first_y_data = pd.read_csv(first_y_file_path, encoding='gbk')
        except UnicodeDecodeError:
            first_y_data = pd.read_csv(first_y_file_path, encoding='latin1')
    
    # 为了进行聚类分析，我们需要从first_y_data中提取与之前聚类相同的特征
    clustering_features = ['孕妇BMI', 'Y染色体浓度']
    first_y_X = first_y_data[clustering_features].dropna()
    
    # 对first_y_data进行聚类预测
    first_y_clusters = kmeans.predict(first_y_X)
    
    # 将聚类结果添加到数据中
    first_y_X_with_clusters = first_y_X.copy()
    first_y_X_with_clusters['cluster'] = first_y_clusters
    
    # 根据聚类结果分离不同类别的数据
    clusters = {}
    for i in range(kmeans.n_clusters):
        clusters[i] = first_y_X_with_clusters[first_y_X_with_clusters['cluster'] == i]
    
    print("\n=== 根据聚类结果分类的数据统计 ===")
    for i in range(kmeans.n_clusters):
        print(f"簇 {i}: {len(clusters[i])} 个样本")
    
    # 计算各类别中检测孕周的90%分位数作为最佳时点
    # 注意：first_y_data中需要有检测孕周这一列
    first_y_data_with_clusters = first_y_data.loc[first_y_X.index].copy()
    first_y_data_with_clusters['cluster'] = first_y_clusters
    
    print("\n=== 各聚类类别90%分位孕周 ===")
    cluster_week_90 = {}
    for i in range(kmeans.n_clusters):
        cluster_data = first_y_data_with_clusters[first_y_data_with_clusters['cluster'] == i]
        if len(cluster_data) > 0:
            week_90 = np.percentile(cluster_data['检测孕周'], 10)
            cluster_week_90[i] = week_90
            print(f"簇 {i} 的90%分位孕周: {format_gestational_age(week_90)} ({week_90:.2f}天)")
    
    # 取总体90%分位孕周作为最佳时点
    overall_week_90 = np.percentile(first_y_data_with_clusters['检测孕周'], 10)
    print(f"\n总体90%分位孕周: {format_gestational_age(overall_week_90)} ({overall_week_90:.2f}天)")
    
    # 可视化各类别的孕周分布
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, kmeans.n_clusters))
    
    for i in range(kmeans.n_clusters):
        cluster_data = first_y_data_with_clusters[first_y_data_with_clusters['cluster'] == i]
        if len(cluster_data) > 0:
            plt.hist(cluster_data['检测孕周'], bins=20, alpha=0.7, color=colors[i], label=f'簇 {i}')
    
    plt.axvline(overall_week_90, color='black', linestyle='--', linewidth=2, 
                label=f'总体90%分位 ({format_gestational_age(overall_week_90)})')
    plt.xlabel('检测孕周 (天)')
    plt.ylabel('频数')
    plt.title('不同聚类类别下检测孕周的分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 显示每个簇的特征描述
    print("\n=== 各簇特征描述 ===")
    for i in range(kmeans.n_clusters):
        cluster_data = first_y_data_with_clusters[first_y_data_with_clusters['cluster'] == i]
        if len(cluster_data) > 0:
            print(f"\n簇 {i}:")
            print(f"  孕妇BMI范围: [{cluster_data['孕妇BMI'].min():.2f}, {cluster_data['孕妇BMI'].max():.2f}]")
            print(f"  孕妇BMI均值: {cluster_data['孕妇BMI'].mean():.2f}")
            print(f"  Y染色体浓度范围: [{cluster_data['Y染色体浓度'].min():.2f}, {cluster_data['Y染色体浓度'].max():.2f}]")
            print(f"  Y染色体浓度均值: {cluster_data['Y染色体浓度'].mean():.2f}")
            
            # 格式化显示孕周信息
            min_weeks, min_days = days_to_weeks_days(cluster_data['检测孕周'].min())
            max_weeks, max_days = days_to_weeks_days(cluster_data['检测孕周'].max())
            mean_weeks, mean_days = days_to_weeks_days(cluster_data['检测孕周'].mean())
            
            print(f"  检测孕周范围: [{format_gestational_age(cluster_data['检测孕周'].min())}, {format_gestational_age(cluster_data['检测孕周'].max())}]")
            print(f"  检测孕周均值: {format_gestational_age(cluster_data['检测孕周'].mean())}")