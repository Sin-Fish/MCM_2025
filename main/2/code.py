import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 聚类数量配置
N_CLUSTERS = 4

check_time = '检测孕周'

# 修复路径问题 - 使用相对路径添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.util.data_manager import Data
from model.k_means import KMeansCluster
from model.km_model import KMModel
from data.util.draw import draw_col_seaborn
from model.logistic_regression import LogisticRegressionModel
from main.util.diagnostic_efficacy_ratio import diagnostic_efficacy_ratio


def estimate_survival_times(survival_function, format_func):
    """
    通过插值估算中位生存时间和诊断效率比最大值对应的时间点
    
    参数:
    survival_function: 生存函数
    format_func: 时间格式化函数
    
    返回:
    中位生存时间和诊断效率比最大值时间点
    """
    if not hasattr(survival_function, 'iloc'):
        return None, None
    
    # 获取曲线数据
    times = survival_function.index
    surv_probs = survival_function.iloc[:, 0]
    
    # 确保概率是递减的
    if len(surv_probs) > 1 and surv_probs.iloc[0] < surv_probs.iloc[-1]:
        return None, None
    
    # 估算中位生存时间 (生存概率=0.5)
    median_time = None
    max_diagnostic_ratio_time = None
    max_diagnostic_ratio = -1  # 初始化为负值，确保能找到最大值
    
    # 查找中位生存时间
    if surv_probs.min() <= 0.5 <= surv_probs.max():
        median_idx = np.argmax(surv_probs <= 0.5)
        if median_idx > 0:
            # 线性插值计算中位时间
            t1, t2 = times[median_idx-1], times[median_idx]
            p1, p2 = surv_probs.iloc[median_idx-1], surv_probs.iloc[median_idx]
            median_time = t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1)
    
    # 查找诊断效率比最大值对应的时间点（排除时间0点）
    for i in range(len(times)):
        time_point = times[i]
        # 跳过时间0点，因为该点的生存概率为1.0，且风险因子计算可能不准确
        if time_point <= 0:
            continue
        survival_prob = surv_probs.iloc[i]
        # 计算诊断效率比
        ratio = diagnostic_efficacy_ratio(time_point, 1-survival_prob)
        if ratio > max_diagnostic_ratio:
            max_diagnostic_ratio = ratio
            max_diagnostic_ratio_time = time_point
    
    return median_time, max_diagnostic_ratio_time


def analyze_km_results(km_fitters, format_func):
    """
    分析KM模型结果，计算中位生存时间和诊断效率比最大值时间点
    
    参数:
    km_fitters: KM拟合器字典
    format_func: 时间格式化函数
    
    返回:
    分析结果字典
    """
    results = {}
    
    for cluster_id, fitter in km_fitters.items():
        print(f"\n=== 簇 {cluster_id} 的KM分析结果 ===")
        
        # 估算中位生存时间和诊断效率比最大值时间点
        median_time, max_diagnostic_ratio_time = estimate_survival_times(
            fitter.survival_function_, format_func
        )
        
        if median_time is not None:
            print(f"簇 {cluster_id} 中位生存时间: {format_func(median_time)} (原始值: {median_time:.2f}天)")
        else:
            print(f"簇 {cluster_id} 中位生存时间: 无法估算")
            
        if max_diagnostic_ratio_time is not None:
            print(f"簇 {cluster_id} 最佳诊断效率比时间点: {format_func(max_diagnostic_ratio_time)} (原始值: {max_diagnostic_ratio_time:.2f}天)")
            # 计算该时间点的诊断效率比值
            # 需要从生存函数中获取该时间点的生存概率
            survival_prob = np.interp(max_diagnostic_ratio_time, 
                                    fitter.survival_function_.index, 
                                    fitter.survival_function_.iloc[:, 0])
            ratio = diagnostic_efficacy_ratio(max_diagnostic_ratio_time, survival_prob)
            print(f"簇 {cluster_id} 最佳诊断效率比值: {ratio:.4f}")
        else:
            print(f"簇 {cluster_id} 最佳诊断效率比时间点: 无法估算")
        
        results[cluster_id] = {
            'median_time': median_time,
            'max_diagnostic_ratio_time': max_diagnostic_ratio_time
        }
     
    return results


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


def perform_kmeans_analysis(data, n_clusters=4):
    """
    执行K-means聚类分析
    
    参数:
    data: 原始数据
    n_clusters: 聚类数量
    
    返回:
    kmeans模型和聚类结果
    """
    # 特征选择
    feature = ['孕妇BMI', "Y染色体浓度"]
    X = data[feature].dropna()
    
    # 训练K-means模型
    kmeans = KMeansCluster(n_clusters=n_clusters)
    kmeans.train(X)
    
    # 重新映射聚类标签，使标签按聚类中心数值递增排序
    # 获取聚类中心
    centers = kmeans.get_cluster_centers()
    
    # 根据第一个特征（孕妇BMI）对标签进行排序
    sorted_indices = np.argsort(centers[:, 0])
    
    # 创建标签映射
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
    
    # 重新映射训练数据的标签
    remapped_labels = np.array([label_mapping[label] for label in kmeans.get_labels()])
    kmeans.labels_ = remapped_labels
    
    # 重新映射聚类中心顺序
    remapped_centers = centers[sorted_indices]
    kmeans.cluster_centers_ = remapped_centers
    
    # 更新模型的预测方法，使其使用新的标签映射
    original_predict = kmeans.model.predict
    def remapped_predict(X):
        original_labels = original_predict(X)
        return np.array([label_mapping[label] for label in original_labels])
    
    kmeans.model.predict = remapped_predict
    
    # 输出聚类信息
    print("聚类中心:")
    print(kmeans.get_cluster_centers())
    print("惯性 (簇内平方和):", kmeans.get_inertia())
    print("前10个样本的聚类标签:", kmeans.get_labels()[:10])
    
    # 输出边界信息
    if len(feature) == 2:
        kmeans.print_cluster_boundaries(X, feature[0], feature[1])
    elif len(feature) == 1:
        kmeans.print_cluster_boundaries(X, feature[0])
    
    return kmeans, X, feature


def perform_km_analysis(data, kmeans_model, feature):
    """
    执行Kaplan-Meier生存分析（使用新模型）
    
    参数:
    data: 原始数据
    kmeans_model: 训练好的K-means模型
    feature: 用于聚类的特征列表
    
    返回:
    KM分析结果
    """
    # 准备用于Kaplan-Meier分析的数据
    km_data = prepare_pregnancy_data_for_km(data)
    
    # 根据BMI进行聚类分组（使用与之前训练聚类模型时相同的特征）
    km_X = km_data[feature].dropna()  # 使用相同的特征列名
    km_data_clean = km_data.loc[km_X.index].copy()
    
    # 对数据进行聚类预测
    km_clusters = kmeans_model.predict(km_X)
    km_data_clean['cluster'] = km_clusters
    
    print("\n=== 根据聚类结果分类的数据统计 ===")
    for i in range(kmeans_model.n_clusters):
        cluster_count = len(km_data_clean[km_data_clean['cluster'] == i])
        print(f"簇 {i}: {cluster_count} 个孕妇")
    
    # 使用新的KM模型进行分析
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 为每个簇拟合KM曲线
    km_models = {}
    km_fitters = {}  # 用于兼容旧的分析方法
    
    for cluster_id in range(kmeans_model.n_clusters):
        cluster_data = km_data_clean[km_data_clean['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            # 创建KM模型实例
            km_models[cluster_id] = KMModel()
            
            # 拟合数据
            km_models[cluster_id].fit(
                cluster_data['事件时间'], 
                cluster_data['事件发生'], 
                label=f'簇 {cluster_id}'
            )
            
            # 为了兼容旧的分析方法，我们保留原始的KaplanMeierFitter实例
            km_fitters[cluster_id] = km_models[cluster_id].model
            
            # 绘制曲线
            km_models[cluster_id].plot(ax=ax, ci_show=False)
    
    # 添加10%的参考线
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='生存概率中位线')
    
    ax.set_xlabel(check_time)
    ax.set_ylabel('未达标概率')
    ax.set_title('各簇的Kaplan-Meier未达标概率曲线')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # 使用KM模型中的分析方法计算结果
    print("\n=== 各簇Kaplan-Meier分析结果 ===")
    km_model_analyzer = KMModel()
    results = analyze_km_results(km_fitters, format_gestational_age)
    
    # 统一输出各簇的中位生存时间和诊断效率比最大值时间点，方便比对
    print("\n=== KM模型分析结果 ===")

    for cluster_id in sorted(results.keys()):
        result = results[cluster_id]
        median_time = result['median_time']
        max_diagnostic_ratio_time = result['max_diagnostic_ratio_time']
        
        if median_time is not None:
            print(f"簇 {cluster_id} 中位生存时间: {format_gestational_age(median_time)} (原始值: {median_time:.2f}天)")
        else:
            print(f"簇 {cluster_id} 中位生存时间: 无法估算")
            
        if max_diagnostic_ratio_time is not None:
            print(f"簇 {cluster_id} 最佳诊断效率比时间点: {format_gestational_age(max_diagnostic_ratio_time)} (原始值: {max_diagnostic_ratio_time:.2f}天)")
        else:
            print(f"簇 {cluster_id} 最佳诊断效率比时间点: 无法估算")
    
    return results


def main_analysis_pipeline(data):
    """
    主分析流程函数，整合聚类分析和KM分析
    
    参数:
    data: 原始数据
    
    返回:
    分析结果
    """
    print("开始执行主分析流程...")
    
    # 执行K-means聚类分析
    kmeans_model, X, feature = perform_kmeans_analysis(data, n_clusters=N_CLUSTERS)
    
    # 执行KM生存分析
    km_results = perform_km_analysis(data, kmeans_model, feature)
    
    print("\n主分析流程执行完成。")
    return {
        'kmeans_model': kmeans_model,
        'km_results': km_results
    }


if __name__ == "__main__":
    data = Data.data
    
    # 执行主分析流程
    analysis_results = main_analysis_pipeline(data)
