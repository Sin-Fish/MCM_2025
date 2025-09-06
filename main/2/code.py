import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


check_time = '检测孕周'


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.util.data_manager import Data
from model.k_means import KMeansCluster
from data.util.draw import draw_col_seaborn
from model.logistic_regression import LogisticRegressionModel

def plot_cluster_data(cluster_id, data, title):
    """绘制簇数据的散点图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data[check_time], data['Y染色体浓度'], alpha=0.7)
    plt.xlabel(check_time)
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

def prepare_pregnancy_data_for_km(data):
    """
    为Kaplan-Meier分析准备孕妇数据
    根据Y染色体浓度是否达标(>=0.04)来确定事件时间
    """
    # Y染色体浓度阈值
    Y_THRESHOLD = 0.04
    
    # 按孕妇代码分组
    grouped = data.groupby('孕妇代码')
    
    results = []
    
    for woman_id, group in grouped:
        # 按检测孕周排序
        group = group.sort_values(check_time)
        
        # 找到所有达标检测（Y染色体浓度 >= 0.04）
        qualified_tests = group[group['Y染色体浓度'] >= Y_THRESHOLD]
        
        # 找到所有不达标检测（Y染色体浓度 < 0.04）
        unqualified_tests = group[group['Y染色体浓度'] < Y_THRESHOLD]
        
        # 如果有达标检测
        if len(qualified_tests) > 0:
            # 第一次达标时间
            first_qualified_time = qualified_tests.iloc[0][check_time]
            
            # 添加记录：事件发生（达标）
            results.append({
                '孕妇代码': woman_id,
                '事件时间': first_qualified_time,
                '事件发生': 1,  # 事件发生（达标）
                '孕妇BMI': group.iloc[0]['孕妇BMI'],
                'Y染色体浓度': group.iloc[0]['Y染色体浓度']  # 添加Y染色体浓度用于聚类
            })
        else:
            # 如果没有达标检测，使用最后一次检测时间作为删失时间
            if len(group) > 0:
                last_test_time = group.iloc[-1][check_time]
                results.append({
                    '孕妇代码': woman_id,
                    '事件时间': last_test_time,
                    '事件发生': 0,  # 删失（未达标）
                    '孕妇BMI': group.iloc[0]['孕妇BMI'],
                    'Y染色体浓度': group.iloc[0]['Y染色体浓度']  # 添加Y染色体浓度用于聚类
                })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    data = Data.data
    # 统计每个孕妇达标孕周，
    # feature = [check_time, '孕妇BMI']
    feature = ['孕妇BMI', "Y染色体浓度"]
    
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

    # 准备用于Kaplan-Meier分析的数据
    km_data = prepare_pregnancy_data_for_km(data)
    
    # 根据BMI进行聚类分组（使用与之前训练聚类模型时相同的特征）
    km_X = km_data[feature].dropna()  # 使用相同的特征列名
    km_data_clean = km_data.loc[km_X.index].copy()
    
    # 对数据进行聚类预测
    km_clusters = kmeans.predict(km_X)
    km_data_clean['cluster'] = km_clusters
    
    print("\n=== 根据聚类结果分类的数据统计 ===")
    for i in range(kmeans.n_clusters):
        cluster_count = len(km_data_clean[km_data_clean['cluster'] == i])
        print(f"簇 {i}: {cluster_count} 个孕妇")
    
    # 使用lifelines库绘制Kaplan-Meier曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 为每个簇绘制KM曲线
    km_fitters = {}
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = km_data_clean[km_data_clean['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            # 创建KaplanMeierFitter实例
            kmf = KaplanMeierFitter()
            km_fitters[cluster_id] = kmf
            
            # 拟合数据
            kmf.fit(cluster_data['事件时间'], event_observed=cluster_data['事件发生'], label=f'簇 {cluster_id}')
            
            # 绘制曲线
            kmf.plot_survival_function(ax=ax)
    
    # 添加10%的参考线
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='10%未达标参考线')
    
    ax.set_xlabel(check_time)
    ax.set_ylabel('未达标概率')
    ax.set_title('各簇的Kaplan-Meier未达标概率曲线')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # 计算各簇的中位生存时间和90%分位数时间
    print("\n=== 各簇Kaplan-Meier分析结果 ===")
    for cluster_id in range(kmeans.n_clusters):
        if cluster_id in km_fitters:
            kmf = km_fitters[cluster_id]
            median_time = kmf.median_survival_time_
            if not np.isnan(median_time):
                print(f"簇 {cluster_id} 中位生存时间: {format_gestational_age(median_time)} (原始值: {median_time:.2f}天)")
            else:
                print(f"簇 {cluster_id} 中位生存时间: 无法计算（数据不足）")
            
            # 计算90%分位数时间（即10%的孕妇仍未达标的时间）
            # 这里我们计算生存概率降到0.1的时间点
            try:
                # 获取生存概率和时间
                survival_prob = kmf.survival_function_.values.flatten()
                times = kmf.survival_function_.index
                
                # 找到生存概率降到0.1的时间点
                # 注意：KM曲线是递减的，所以我们找第一个小于等于0.1的点
                below_10pct = survival_prob <= 0.1
                if np.any(below_10pct):
                    idx = np.where(below_10pct)[0][0]
                    time_10pct = times[idx]
                    print(f"簇 {cluster_id} 10%未达标时间: {format_gestational_age(time_10pct)} (原始值: {time_10pct:.2f}天)")
                else:
                    print(f"簇 {cluster_id} 10%未达标时间: 超出观察范围")
            except Exception as e:
                print(f"簇 {cluster_id} 10%未达标时间: 计算失败 ({e})")