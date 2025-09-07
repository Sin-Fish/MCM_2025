import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.util.data_manager import Data
from model.k_means import KMeansCluster
from model.cox import CoxModel
from main.util.monte_carlo_test import MonteCarloFramework

class CoxModelAdapter:
    """适配器类，使CoxModel与蒙特卡洛框架兼容"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def fit(self, X, y):
        """训练模型 - Cox模型需要特殊处理"""
        # Cox模型在蒙特卡洛框架中将在每次模拟时单独训练
        pass
        
    def predict(self, X):
        """预测 - Cox模型主要用于生存分析，此处返回默认值"""
        # Cox模型主要用于生存分析，而非直接预测
        return np.zeros(len(X))


def prepare_pregnancy_data_for_cox_with_noise(data, sigma_y=0.1, sigma_time=0.1):
    """
    为Cox比例风险模型分析准备带噪声的孕妇数据
    根据Y染色体浓度是否达标(>=0.04)来确定事件时间
    """
    # Y染色体浓度阈值
    Y_THRESHOLD = 0.04
    
    # 向Y染色体浓度添加噪声
    noisy_data = data.copy()
    if sigma_y > 0:
        noisy_y_concentration = np.random.normal(data['Y染色体浓度'].values, sigma_y)
        # 确保噪声后的Y染色体浓度不为负数
        noisy_y_concentration = np.maximum(noisy_y_concentration, 0)
        noisy_data['Y染色体浓度'] = noisy_y_concentration
    
    # 向检测时间添加噪声（如果sigma_time > 0）
    if sigma_time > 0:
        noisy_time = np.random.normal(data['检测孕周'].values, sigma_time)
        # 确保噪声后的时间不为负数
        noisy_time = np.maximum(noisy_time, 0)
        noisy_data['检测孕周'] = noisy_time
    
    # 按孕妇代码分组
    grouped = noisy_data.groupby('孕妇代码')
    
    results = []
    
    for woman_id, group in grouped:
        # 按检测孕周排序
        group = group.sort_values('检测孕周')
        
        # 找到所有达标检测（Y染色体浓度 >= 0.04）
        qualified_tests = group[group['Y染色体浓度'] >= Y_THRESHOLD]
        
        # 如果有达标检测
        if len(qualified_tests) > 0:
            # 第一次达标时间
            first_qualified_time = qualified_tests.iloc[0]['检测孕周']
            
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
                'Y染色体浓度': group.iloc[0]['Y染色体浓度'],  # 添加Y染色体浓度用于聚类
                '孕妇BMI': group.iloc[0]['孕妇BMI'] if '孕妇BMI' in group.columns else np.nan,  # 添加孕妇BMI用于聚类
                'Y染色体的Z值': group.iloc[0]['Y染色体的Z值'] if 'Y染色体的Z值' in group.columns else np.nan,
                '被过滤掉读段数的比例': group.iloc[0]['被过滤掉读段数的比例'] if '被过滤掉读段数的比例' in group.columns else np.nan,
                'GC含量': group.iloc[0]['GC含量'] if 'GC含量' in group.columns else np.nan,
                '身高': group.iloc[0]['身高'] if '身高' in group.columns else np.nan,
                '体重': group.iloc[0]['体重'] if '体重' in group.columns else np.nan
            })
        else:
            # 如果没有达标检测，使用最后一次检测时间作为删失时间
            if len(group) > 0:
                last_test_time = group.iloc[-1]['检测孕周']
                results.append({
                    '孕妇代码': woman_id,
                    '事件时间': last_test_time,
                    '事件发生': 0,  # 删失（未达标）
                    '年龄': group.iloc[0]['年龄'] if '年龄' in group.columns else np.nan,
                    '检测抽血次数': len(group),  # 检测抽血次数
                    '检测孕周': group.iloc[0]['检测孕周'],
                    '原始读段数': group.iloc[0]['原始读段数'] if '原始读段数' in group.columns else np.nan,
                    '在参考基因组上比对的比例': group.iloc[0]['在参考基因组上比对的比例'] if '在参考基因组上比对的比例' in group.columns else np.nan,
                    'Y染色体浓度': group.iloc[0]['Y染色体浓度'],  # 添加Y染色体浓度用于聚类
                    '孕妇BMI': group.iloc[0]['孕妇BMI'] if '孕妇BMI' in group.columns else np.nan,  # 添加孕妇BMI用于聚类
                    'Y染色体的Z值': group.iloc[0]['Y染色体的Z值'] if 'Y染色体的Z值' in group.columns else np.nan,
                    '被过滤掉读段数的比例': group.iloc[0]['被过滤掉读段数的比例'] if '被过滤掉读段数的比例' in group.columns else np.nan,
                    'GC含量': group.iloc[0]['GC含量'] if 'GC含量' in group.columns else np.nan,
                    '身高': group.iloc[0]['身高'] if '身高' in group.columns else np.nan,
                    '体重': group.iloc[0]['体重'] if '体重' in group.columns else np.nan
                })
    
    return pd.DataFrame(results)


def perform_cox_analysis_for_monte_carlo(data, feature_columns):
    """
    为蒙特卡洛测试执行Cox分析
    
    参数:
    data: 准备好的数据
    feature_columns: 特征列名列表
    
    返回:
    分析结果，包括各簇的中位时间和90%分位时间
    """
    # 定义聚类特征
    cluster_features = ['孕妇BMI', 'Y染色体浓度']
    
    # 获取聚类特征数据
    cluster_X = data[cluster_features].dropna()
    data_clean = data.loc[cluster_X.index].copy()
    
    # 训练K-means模型
    kmeans = KMeansCluster(n_clusters=4)
    kmeans.train(cluster_X)
    
    # 对数据进行聚类预测
    clusters = kmeans.predict(cluster_X)
    data_clean['cluster'] = clusters
    
    # 为每个簇计算分位时间
    percentile_times = {}
    
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = data_clean[data_clean['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            # 计算50%和90%分位时间
            if len(cluster_data[cluster_data['事件发生'] == 1]) > 0:  # 确保有事件发生
                percentile_50 = np.percentile(cluster_data['事件时间'], 50)
                percentile_90 = np.percentile(cluster_data['事件时间'], 90)
                percentile_times[cluster_id] = {
                    '50': percentile_50,
                    '90': percentile_90
                }
    
    return percentile_times


def monte_carlo_with_cox_analysis(original_data, n_simulations=100, sigma_y=0.1, sigma_time=0.1):
    """
    使用Cox模型进行蒙特卡洛模拟
    
    参数:
    original_data: 原始数据
    n_simulations: 模拟次数
    sigma_y: Y染色体浓度噪声标准差
    sigma_time: 检测时间噪声标准差
    
    返回:
    模拟结果统计信息
    """
    # 定义Cox模型使用的特征列
    feature_columns = ['年龄', '检测抽血次数', "检测孕周", '身高', '体重', "Y染色体的Z值"]
    
    # 为每个分位数创建存储列表
    percentile_times_50 = []
    percentile_times_90 = []
    
    print(f"开始进行{n_simulations}次蒙特卡罗模拟，Y浓度噪声σ={sigma_y}，时间噪声σ={sigma_time}")
    
    for i in range(n_simulations):
        if (i + 1) % 20 == 0:
            print(f"已完成 {i + 1}/{n_simulations} 次模拟")
        
        # 生成扰动数据集
        perturbed_data = prepare_pregnancy_data_for_cox_with_noise(
            original_data, sigma_y, sigma_time)
        
        # 执行Cox分析
        try:
            cluster_times = perform_cox_analysis_for_monte_carlo(perturbed_data, feature_columns)
            
            # 收集各簇的时间数据
            cluster_50_times = []
            cluster_90_times = []
            
            for cluster_id, times in cluster_times.items():
                cluster_50_times.append(times['50'])
                cluster_90_times.append(times['90'])
            
            # 计算平均时间
            if cluster_50_times:
                avg_50_time = np.mean(cluster_50_times)
                percentile_times_50.append(avg_50_time)
                
            if cluster_90_times:
                avg_90_time = np.mean(cluster_90_times)
                percentile_times_90.append(avg_90_time)
                
        except Exception as e:
            print(f"第{i+1}次模拟失败: {e}")
            continue
    
    print(f"蒙特卡罗模拟完成，共{len(percentile_times_50)}次有效模拟")
    
    # 分析模拟结果
    if len(percentile_times_50) > 0:
        stats_results = {}
        
        # 50%分位数分析
        mean_50 = np.mean(percentile_times_50)
        std_50 = np.std(percentile_times_50)
        
        print(f"\n=== 50%分位时间分析 (σ_y={sigma_y}, σ_time={sigma_time}) ===")
        print(f"平均值: {mean_50:.2f} 天")
        print(f"标准差: {std_50:.2f} 天")
        
        stats_results[50] = {
            'mean': mean_50,
            'std': std_50
        }
        
        # 90%分位数分析
        mean_90 = np.mean(percentile_times_90)
        std_90 = np.std(percentile_times_90)
        
        print(f"\n=== 90%分位时间分析 (σ_y={sigma_y}, σ_time={sigma_time}) ===")
        print(f"平均值: {mean_90:.2f} 天")
        print(f"标准差: {std_90:.2f} 天")
        
        stats_results[90] = {
            'mean': mean_90,
            'std': std_90
        }
        
        return {
            'percentile_times': {50: percentile_times_50, 90: percentile_times_90},
            'stats': stats_results,
            'n_valid_simulations': len(percentile_times_50)
        }
    else:
        print("没有有效的模拟结果")
        return None


def test_cox_monte_carlo():
    """测试Cox模型与蒙特卡洛框架的集成"""
    print("=== 测试Cox模型与蒙特卡洛框架的集成 ===")
    
    # 加载数据
    data = Data.data
    
    # 使用适配器创建蒙特卡洛框架实例
    cox_adapter = CoxModelAdapter()
    kmeans_model = KMeansCluster(n_clusters=4)
    
    mc_framework = MonteCarloFramework(
        model=cox_adapter,
        cluster_model=kmeans_model,
        feature_columns=['孕妇BMI', 'Y染色体浓度']
    )
    
    # 运行敏感性分析
    sensitivity_results = mc_framework.run_sensitivity_analysis(
        original_data=data,
        sigma_y_values=[0.05, 0.1],
        sigma_time_values=[2.0, 5.0],
        n_simulations=50,  # 减少模拟次数以加快测试速度
        percentiles=[50, 90],
        visualize=False
    )
    
    print("\n=== 敏感性分析结果 ===")
    # 显示部分结果
    for key, result in list(sensitivity_results.items())[:2]:
        print(f"{key}: {result.get('n_valid_simulations', 'N/A')} 次有效模拟")
    
    return sensitivity_results


def test_direct_cox_monte_carlo():
    """直接测试Cox模型的蒙特卡洛分析"""
    print("\n=== 直接测试Cox模型的蒙特卡洛分析 ===")
    
    # 加载数据
    data = Data.data
    
    # 运行蒙特卡洛模拟
    result = monte_carlo_with_cox_analysis(
        data, 
        n_simulations=30,  # 减少模拟次数以加快测试速度
        sigma_y=0.1, 
        sigma_time=2.0
    )
    
    if result:
        print(f"模拟完成，有效模拟次数: {result['n_valid_simulations']}")
        print(f"50%分位时间: 均值={result['stats'][50]['mean']:.2f}±{result['stats'][50]['std']:.2f}天")
        print(f"90%分位时间: 均值={result['stats'][90]['mean']:.2f}±{result['stats'][90]['std']:.2f}天")
    
    return result


if __name__ == "__main__":
    print("开始测试Cox模型与蒙特卡洛框架的集成...")
    
    # 测试使用新框架
    framework_results = test_cox_monte_carlo()
    
    # 测试直接实现
    direct_results = test_direct_cox_monte_carlo()
    
    print("\n测试完成！")