import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

check_time = '检测孕周'


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.util.data_manager import Data
from model.k_means import KMeansCluster
from model.cox import CoxModel  
from data.util.draw import draw_col_seaborn, draw_col_hot_map_seaborn


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

def prepare_pregnancy_data_for_cox(data):
    """
    为Cox比例风险模型分析准备孕妇数据
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
                '年龄': group.iloc[0]['年龄'] if '年龄' in group.columns else np.nan,
                '检测抽血次数': len(group),  # 检测抽血次数
                '检测孕周': group.iloc[0]['检测孕周'],
                '原始读段数': group.iloc[0]['原始读段数'] if '原始读段数' in group.columns else np.nan,
                '在参考基因组上比对的比例': group.iloc[0]['在参考基因组上比对的比例'] if '在参考基因组上比对的比例' in group.columns else np.nan,
                'Y染色体的Z值': group.iloc[0]['Y染色体的Z值'] if 'Y染色体的Z值' in group.columns else np.nan,
                'Y染色体浓度': group.iloc[0]['Y染色体浓度'],  # 添加Y染色体浓度用于聚类
                '孕妇BMI': group.iloc[0]['孕妇BMI'] if '孕妇BMI' in group.columns else np.nan  # 添加孕妇BMI用于聚类
            })
        else:
            # 如果没有达标检测，使用最后一次检测时间作为删失时间
            if len(group) > 0:
                last_test_time = group.iloc[-1][check_time]
                results.append({
                    '孕妇代码': woman_id,
                    '事件时间': last_test_time,
                    '事件发生': 0,  # 删失（未达标）
                    '年龄': group.iloc[0]['年龄'] if '年龄' in group.columns else np.nan,
                    '检测抽血次数': len(group),  # 检测抽血次数
                    '检测孕周': group.iloc[0]['检测孕周'],
                    '原始读段数': group.iloc[0]['原始读段数'] if '原始读段数' in group.columns else np.nan,
                    '在参考基因组上比对的比例': group.iloc[0]['在参考基因组上比对的比例'] if '在参考基因组上比对的比例' in group.columns else np.nan,
                'Y染色体的Z值': group.iloc[0]['Y染色体的Z值'] if 'Y染色体的Z值' in group.columns else np.nan,
                    'Y染色体浓度': group.iloc[0]['Y染色体浓度'],  # 添加Y染色体浓度用于聚类
                    '孕妇BMI': group.iloc[0]['孕妇BMI'] if '孕妇BMI' in group.columns else np.nan  # 添加孕妇BMI用于聚类
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
    # 特征选择 - 仅使用孕妇BMI
    feature = ['孕妇BMI']
    X = data[feature].dropna()
    
    # 训练K-means模型
    kmeans = KMeansCluster(n_clusters=n_clusters)
    kmeans.train(X)
    
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

def plot_cox_coefficients_heatmap(cox_models):
    """
    绘制Cox模型协变量影响系数热力图
    
    参数:
    cox_models: Cox模型字典，键为簇ID，值为训练好的CoxModel实例
    """
    # 收集所有模型的系数
    coefficients_data = {}
    for cluster_id, model in cox_models.items():
        if hasattr(model, 'get_coefficients'):
            try:
                coeffs = model.get_coefficients()
                coefficients_data[f'簇 {cluster_id}'] = coeffs
            except:
                continue
    
    if not coefficients_data:
        print("没有可用的系数数据用于绘制热力图")
        return
    
    # 构建系数矩阵
    all_features = set()
    for coeffs in coefficients_data.values():
        all_features.update(coeffs.index)
    
    all_features = sorted(list(all_features))
    
    # 创建系数矩阵
    coef_matrix = pd.DataFrame(index=all_features)
    for cluster_name, coeffs in coefficients_data.items():
        coef_matrix[cluster_name] = coeffs.reindex(all_features, fill_value=0)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(coef_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.3f', cbar_kws={'label': '系数值'})
    plt.title('各簇Cox模型协变量影响系数热力图')
    plt.xlabel('簇')
    plt.ylabel('协变量')
    plt.tight_layout()
    plt.show()

def estimate_survival_times(survival_function, format_func):
    """
    通过插值估算中位生存时间和10%未达标时间
    
    参数:
    survival_function: 生存函数
    format_func: 时间格式化函数
    
    返回:
    中位生存时间和10%未达标时间
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
    ten_percent_time = None
    
    # 查找中位生存时间
    if surv_probs.min() <= 0.5 <= surv_probs.max():
        median_idx = np.argmax(surv_probs <= 0.5)
        if median_idx > 0:
            # 线性插值计算中位时间
            t1, t2 = times[median_idx-1], times[median_idx]
            p1, p2 = surv_probs.iloc[median_idx-1], surv_probs.iloc[median_idx]
            median_time = t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1)
    
    # 查找10%未达标时间 (生存概率=0.1)
    if surv_probs.min() <= 0.1 <= surv_probs.max():
        ten_percent_idx = np.argmax(surv_probs <= 0.1)
        if ten_percent_idx > 0:
            # 线性插值计算10%未达标时间
            t1, t2 = times[ten_percent_idx-1], times[ten_percent_idx]
            p1, p2 = surv_probs.iloc[ten_percent_idx-1], surv_probs.iloc[ten_percent_idx]
            ten_percent_time = t1 + (0.1 - p1) * (t2 - t1) / (p2 - p1)
    
    return median_time, ten_percent_time

def analyze_cox_results(cox_models, format_func, survival_functions):
    """
    分析Cox模型结果，计算中位生存时间和10%未达标时间
    
    参数:
    cox_models: Cox模型字典
    format_func: 时间格式化函数
    survival_functions: 各簇的生存函数
    
    返回:
    分析结果字典
    """
    results = {}
    
    for cluster_id, model in cox_models.items():
        print(f"\n=== 簇 {cluster_id} 的Cox模型分析结果 ===")
        
        try:
            # 输出模型摘要
            if hasattr(model, 'print_summary'):
                model.print_summary()
                
            # 估算中位生存时间和10%未达标时间
            if cluster_id in survival_functions:
                median_time, ten_percent_time = estimate_survival_times(
                    survival_functions[cluster_id], format_func
                )
                
                if median_time is not None:
                    print(f"簇 {cluster_id} 中位生存时间: {format_func(median_time)} (原始值: {median_time:.2f}天)")
                else:
                    print(f"簇 {cluster_id} 中位生存时间: 无法估算")
                    
                if ten_percent_time is not None:
                    print(f"簇 {cluster_id} 10%未达标时间: {format_func(ten_percent_time)} (原始值: {ten_percent_time:.2f}天)")
                else:
                    print(f"簇 {cluster_id} 10%未达标时间: 无法估算")
                
                results[cluster_id] = {
                    'median_time': median_time,
                    'ten_percent_time': ten_percent_time
                }
            else:
                print(f"簇 {cluster_id} 中位生存时间: 无法估算")
                print(f"簇 {cluster_id} 10%未达标时间: 无法估算")
                results[cluster_id] = {
                    'median_time': None,
                    'ten_percent_time': None
                }
                
        except Exception as e:
            print(f"分析簇 {cluster_id} 时出错: {e}")
            results[cluster_id] = {
                'median_time': None,
                'ten_percent_time': None
            }
     
    return results

def perform_cox_analysis(data, kmeans_model, feature):
    """
    执行Cox比例风险模型分析（使用新模型）
    
    参数:
    data: 原始数据
    kmeans_model: 训练好的K-means模型
    feature: 用于聚类的特征列表
    
    返回:
    Cox分析结果
    """
    # 准备用于Cox比例风险模型分析的数据
    cox_data = prepare_pregnancy_data_for_cox(data)
    
    print(f"\n=== 准备Cox分析的数据 ===")
    print(f"总数据量: {len(cox_data)}")
    print(f"事件发生数量: {cox_data['事件发生'].sum()}")
    print(f"删失数量: {len(cox_data) - cox_data['事件发生'].sum()}")
    print(f"事件发生率: {cox_data['事件发生'].mean():.2%}")
    
    # 根据BMI进行聚类分组（使用与之前训练聚类模型时相同的特征）
    cox_X = cox_data[feature].dropna()  # 使用相同的特征列名
    cox_data_clean = cox_data.loc[cox_X.index].copy()
    
    # 对数据进行聚类预测
    cox_clusters = kmeans_model.predict(cox_X)
    cox_data_clean['cluster'] = cox_clusters
    
    print("\n=== 根据聚类结果分类的数据统计 ===")
    for i in range(kmeans_model.n_clusters):
        cluster_data = cox_data_clean[cox_data_clean['cluster'] == i]
        cluster_count = len(cluster_data)
        event_count = cluster_data['事件发生'].sum()
        print(f"簇 {i}: {cluster_count} 个孕妇, {event_count} 个事件")

    # 使用新的Cox模型进行分析
    # 选择新的特征列（不包括BMI）
    feature_columns = ['年龄', '检测抽血次数', '检测孕周', '原始读段数', 
                      '在参考基因组上比对的比例', 'Y染色体的Z值']
    
    # 为每个簇拟合Cox模型
    cox_models = {}
    survival_functions = {}
    
    # 创建用于绘制生存曲线的图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置统一的时间范围从0到200天
    min_time, max_time = 0, 200
    
    for cluster_id in range(kmeans_model.n_clusters):
        cluster_data = cox_data_clean[cox_data_clean['cluster'] == cluster_id]
        print(f"\n=== 簇 {cluster_id} 的详细信息 ===")
        print(f"簇数据量: {len(cluster_data)}")
        
        if len(cluster_data) > 0:
            # 创建Cox模型实例
            cox_models[cluster_id] = CoxModel()
            
            # 准备特征数据，只选择存在的列
            available_features = [col for col in feature_columns if col in cluster_data.columns]
            print(f"可用特征列: {available_features}")
            
            if not available_features:
                print(f"簇 {cluster_id}: 没有可用的特征列")
                continue
                
            # 构建模型数据
            model_data = cluster_data[available_features + ['事件时间', '事件发生']].copy()
            print(f"模型数据量: {len(model_data)}")
            print(f"事件发生数量: {model_data['事件发生'].sum()}")
            
            # 检查缺失值
            print("缺失值统计:")
            for col in model_data.columns:
                missing_count = model_data[col].isna().sum()
                if missing_count > 0:
                    print(f"  {col}: {missing_count} 个缺失值")
            
            # 删除包含缺失值的行
            model_data = model_data.dropna()
            print(f"删除缺失值后数据量: {len(model_data)}")
            
            if len(model_data) == 0:
                print(f"簇 {cluster_id}: 没有足够的数据进行Cox回归分析")
                continue
                
            if model_data['事件发生'].sum() == 0:
                print(f"簇 {cluster_id}: 没有事件发生，无法进行Cox回归分析")
                continue
                
            if len(model_data) < len(available_features) + 1: 
                print(f"簇 {cluster_id}: 数据点不足（{len(model_data)} < {len(available_features)} + 1),无法进行Cox回归分析")
                continue
            
            # 拟合数据
            try:
                print(f"簇 {cluster_id}: 尝试拟合Cox模型...")
                cox_models[cluster_id].train(model_data, event_col='事件发生', duration_col='事件时间')
                print(f"簇 {cluster_id}: Cox模型拟合成功")
                print(f"\n簇 {cluster_id} 的Cox模型摘要:")
                cox_models[cluster_id].print_summary()
                
                # 预测生存函数
                try:
                    # 使用均值作为代表性数据
                    representative_data = model_data[available_features].mean().to_frame().T
                    survival_function = cox_models[cluster_id].predict_survival_function(representative_data)
                    survival_functions[cluster_id] = survival_function
                    
                    # 绘制生存曲线，保持与KM分析一致
                    # 首先获取时间点和生存概率
                    times = survival_function.index
                    surv_probs = survival_function.iloc[:, 0]
                    
                    # 创建从0开始的时间点，确保曲线从(0, 1.0)开始
                    # 按照要求顺序绘制连接线：0点->曲线开始前一天->曲线起始点
                    if len(times) > 0:
                        start_time = times[0] - 1  # 曲线开始前一天
                        # 构建完整的时间序列：0天 -> 前一天 -> 曲线起始点 -> 曲线其余点
                        plot_times = np.concatenate([[0, start_time, times[0]], times])
                        plot_probs = np.concatenate([[1.0, 1.0, surv_probs.iloc[0]], surv_probs])
                    else:
                        plot_times = np.array([0])
                        plot_probs = np.array([1.0])
                    
                    # 绘制曲线
                    ax.plot(plot_times, plot_probs, label=f'簇 {cluster_id}')
                except Exception as e:
                    print(f"绘制簇 {cluster_id} 的生存曲线时出错: {e}")
                    
            except Exception as e:
                print(f"簇 {cluster_id} 的Cox模型拟合失败: {e}")
    
    # 设置统一的时间轴范围从0到200天
    ax.set_xlim(min_time, max_time)
    # 设置y轴范围从-0.05到1.05，与main/2/code.py保持一致，稍微超出0-1范围以提供更好的可视化效果
    ax.set_ylim(-0.05, 1.05)
    
    # 添加10%的参考线（与KM分析保持一致）
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='10%未达标参考线')
    
    # 添加图表标签和图例
    ax.set_xlabel('时间（孕周）')
    ax.set_ylabel('生存概率')
    ax.set_title('各簇的Cox模型生存函数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 绘制协变量影响系数热力图
    plot_cox_coefficients_heatmap(cox_models)
    
    # 分析Cox模型结果
    print("\n=== 各簇Cox模型分析结果 ===")
    cox_results = analyze_cox_results(cox_models, format_gestational_age, survival_functions)
    
    # 统一输出各簇的中位生存时间和10%未达标时间，方便比对
    print("\n=== 统一输出各簇预测时间 ===")
    for cluster_id in sorted(cox_results.keys()):
        result = cox_results[cluster_id]
        median_time = result['median_time']
        ten_percent_time = result['ten_percent_time']
        
        if median_time is not None:
            print(f"簇 {cluster_id} 中位生存时间: {format_gestational_age(median_time)} (原始值: {median_time:.2f}天)")
        else:
            print(f"簇 {cluster_id} 中位生存时间: 无法估算")
            
        if ten_percent_time is not None:
            print(f"簇 {cluster_id} 10%未达标时间: {format_gestational_age(ten_percent_time)} (原始值: {ten_percent_time:.2f}天)")
        else:
            print(f"簇 {cluster_id} 10%未达标时间: 无法估算")
    
    # 统一输出协变量影响数据
    print("\n=== 统一输出协变量影响数据 ===")
    # 收集所有模型的系数
    coefficients_data = {}
    for cluster_id, model in cox_models.items():
        if hasattr(model, 'get_coefficients'):
            try:
                coeffs = model.get_coefficients()
                coefficients_data[cluster_id] = coeffs
            except:
                continue
    
    if coefficients_data:
        # 构建系数矩阵
        all_features = set()
        for coeffs in coefficients_data.values():
            all_features.update(coeffs.index)
        
        all_features = sorted(list(all_features))
        
        # 创建系数矩阵并打印
        coef_matrix = pd.DataFrame(index=all_features)
        for cluster_id, coeffs in coefficients_data.items():
            coef_matrix[f'簇 {cluster_id}'] = coeffs.reindex(all_features, fill_value=0)
        
        # 打印系数矩阵
        print(coef_matrix.to_string(float_format='%.3f'))
    else:
        print("没有可用的协变量影响数据")
    
    return cox_models

def main_analysis_pipeline(data):
    """
    主分析流程函数，整合聚类分析和Cox分析
    
    参数:
    data: 原始数据
    
    返回:
    分析结果
    """
    print("开始执行主分析流程...")
    
    # 执行K-means聚类分析
    kmeans_model, X, feature = perform_kmeans_analysis(data, n_clusters=4)
    
    # 执行Cox比例风险模型分析
    cox_results = perform_cox_analysis(data, kmeans_model, feature)
    
    print("\n主分析流程执行完成。")
    return {
        'kmeans_model': kmeans_model,
        'cox_results': cox_results
    }

if __name__ == "__main__":
    data = Data.data
    
    # 执行主分析流程
    analysis_results = main_analysis_pipeline(data)