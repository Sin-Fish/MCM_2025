import numpy as np
import pandas as pd
from scipy import stats
import sys
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入中文字体设置方法
from data.util.draw import set_chinese_font
from data.util.data_manager import Data
from model.k_means import KMeansCluster

# 设置中文字体
set_chinese_font()


def add_noise_to_data(data, sigma=0.1):
    """
    向数据添加高斯噪声
    
    参数:
    data: 原始数据数组
    sigma: 噪声标准差
    
    返回:
    添加噪声后的数据
    """
    noise = np.random.normal(0, sigma, size=data.shape)
    return data + noise


def prepare_pregnancy_data_with_noise(original_data, sigma_y=0.1, sigma_time=0.1, check_time='检测孕周'):
    """
    为Kaplan-Meier分析准备带噪声的孕妇数据
    根据Y染色体浓度是否达标(>=0.04)来确定事件时间
    
    参数:
    original_data: 原始数据
    sigma_y: Y染色体浓度噪声标准差
    sigma_time: 检测时间噪声标准差
    check_time: 检查时间列名
    
    返回:
    处理后的数据框
    """
    # Y染色体浓度阈值
    Y_THRESHOLD = 0.04
    
    # 向Y染色体浓度添加噪声
    noisy_data = original_data.copy()
    noisy_y_concentration = add_noise_to_data(original_data['Y染色体浓度'].values, sigma_y)
    # 确保噪声后的Y染色体浓度不为负数
    noisy_y_concentration = np.maximum(noisy_y_concentration, 0)
    noisy_data['Y染色体浓度'] = noisy_y_concentration
    
    # 向检测时间添加噪声（如果sigma_time > 0）
    if sigma_time > 0:
        noisy_time = add_noise_to_data(original_data[check_time].values, sigma_time)
        # 确保噪声后的时间不为负数
        noisy_time = np.maximum(noisy_time, 0)
        noisy_data[check_time] = noisy_time
    
    # 按孕妇代码分组
    grouped = noisy_data.groupby('孕妇代码')
    
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


def monte_carlo_simulation(original_data, n_simulations=100, sigma_y=0.1, sigma_time=0.1, 
                          check_time='检测孕周', percentiles=[50, 90]):
    """
    使用蒙特卡罗方法进行敏感性分析
    
    参数:
    original_data: 原始数据
    n_simulations: 模拟次数
    sigma_y: Y染色体浓度噪声标准差
    sigma_time: 检测时间噪声标准差
    check_time: 检查时间列名
    percentiles: 要分析的分位数列表
    
    返回:
    模拟结果统计信息
    """
    # 为每个分位数创建存储列表
    percentile_times = {p: [] for p in percentiles}
    
    # 获取原始数据中的聚类特征
    feature = ['孕妇BMI', "Y染色体浓度"]
    X_original = original_data[feature].dropna()
    
    # 训练原始聚类模型
    kmeans = KMeansCluster(n_clusters=4)
    kmeans.train(X_original)
    
    print(f"开始进行{n_simulations}次蒙特卡罗模拟，Y浓度噪声σ={sigma_y}，时间噪声σ={sigma_time}")
    print(f"分析分位数: {[f'{p}%' for p in percentiles]}")
    
    for i in range(n_simulations):
        if (i + 1) % 20 == 0:
            print(f"已完成 {i + 1}/{n_simulations} 次模拟")
        
        # 生成扰动数据集
        perturbed_data = prepare_pregnancy_data_with_noise(original_data, sigma_y, sigma_time, check_time)
        
        # 准备用于聚类的数据
        km_X = perturbed_data[feature].dropna()
        if len(km_X) == 0:
            continue
            
        perturbed_data_clean = perturbed_data.loc[km_X.index].copy()
        
        # 对数据进行聚类预测
        try:
            km_clusters = kmeans.predict(km_X)
            perturbed_data_clean['cluster'] = km_clusters
        except Exception as e:
            print(f"第{i+1}次模拟聚类预测失败: {e}")
            continue
        
        # 计算各簇的指定分位时间
        cluster_percentile_times = {p: [] for p in percentiles}
        
        for cluster_id in range(kmeans.n_clusters):
            cluster_data = perturbed_data_clean[perturbed_data_clean['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                # 计算指定分位时间
                for p in percentiles:
                    percentile_time = np.percentile(cluster_data['事件时间'], p)
                    cluster_percentile_times[p].append(percentile_time)
        
        # 如果有有效的簇时间，计算平均值作为本次模拟的结果
        for p in percentiles:
            if cluster_percentile_times[p]:
                avg_percentile_time = np.mean(cluster_percentile_times[p])
                percentile_times[p].append(avg_percentile_time)
    
    print(f"蒙特卡罗模拟完成，共{len(percentile_times[percentiles[0]])}次有效模拟")
    
    # 分析模拟结果
    if len(percentile_times[percentiles[0]]) > 0:
        stats_results = {}
        
        for p in percentiles:
            times = percentile_times[p]
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"\n=== {p}%分位时间分析 (σ_y={sigma_y}, σ_time={sigma_time}) ===")
            print(f"平均值: {mean_time:.2f} 天")
            print(f"标准差: {std_time:.2f} 天")
            
            stats_results[p] = {
                'mean': mean_time,
                'std': std_time
            }
        
        return {
            'percentile_times': percentile_times,
            'stats': stats_results,
            'n_valid_simulations': len(percentile_times[percentiles[0]])
        }
    else:
        print("没有有效的模拟结果")
        return None


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


def prepare_baseline_data(original_data, check_time='检测孕周'):
    """
    准备未扰动的基准数据
    
    参数:
    original_data: 原始数据
    check_time: 检查时间列名
    
    返回:
    基准数据统计结果
    """
    # Y染色体浓度阈值
    Y_THRESHOLD = 0.04
    
    # 按孕妇代码分组
    grouped = original_data.groupby('孕妇代码')
    
    results = []
    
    for woman_id, group in grouped:
        # 按检测孕周排序
        group = group.sort_values(check_time)
        
        # 找到所有达标检测（Y染色体浓度 >= 0.04）
        qualified_tests = group[group['Y染色体浓度'] >= Y_THRESHOLD]
        
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
    
    # 转换为DataFrame
    baseline_data = pd.DataFrame(results)
    
    # 获取原始数据中的聚类特征
    feature = ['孕妇BMI', "Y染色体浓度"]
    X_original = original_data[feature].dropna()
    
    # 训练聚类模型
    kmeans = KMeansCluster(n_clusters=4)
    kmeans.train(X_original)
    
    # 准备用于聚类的数据
    km_X = baseline_data[feature].dropna()
    baseline_data_clean = baseline_data.loc[km_X.index].copy()
    
    # 对数据进行聚类预测
    km_clusters = kmeans.predict(km_X)
    baseline_data_clean['cluster'] = km_clusters
    
    return baseline_data_clean, kmeans


def compare_with_baseline(original_data, simulation_results, percentiles=[50, 90], check_time='检测孕周'):
    """
    将模拟结果与未扰动的基准数据进行比较
    
    参数:
    original_data: 原始数据
    simulation_results: 模拟结果
    percentiles: 分位数列表
    check_time: 检查时间列名
    
    返回:
    比较结果字典
    """
    # 准备基准数据
    baseline_data, kmeans = prepare_baseline_data(original_data, check_time)
    
    print("\n=== 基准数据与模拟结果比较 ===")
    print("分位数\t基准值\t\t模拟均值\t差异\t\t变异系数")
    print("-" * 60)
    
    comparison_results = {}
    
    for p in percentiles:
        # 计算基准数据的分位数
        cluster_percentile_times = []
        for cluster_id in range(kmeans.n_clusters):
            cluster_data = baseline_data[baseline_data['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                percentile_time = np.percentile(cluster_data['事件时间'], p)
                cluster_percentile_times.append(percentile_time)
        
        baseline_value = np.mean(cluster_percentile_times) if cluster_percentile_times else 0
        
        # 获取模拟结果
        if p in simulation_results['stats']:
            sim_mean = simulation_results['stats'][p]['mean']
            sim_std = simulation_results['stats'][p]['std']
            diff = sim_mean - baseline_value
            cv = sim_std / sim_mean if sim_mean != 0 else 0
            
            print(f"{p}%\t{format_gestational_age(baseline_value)} ({baseline_value:.2f})\t"
                  f"{format_gestational_age(sim_mean)} ({sim_mean:.2f})\t"
                  f"{diff:.2f}\t\t{cv:.4f}")
            
            comparison_results[p] = {
                'baseline_value': baseline_value,
                'baseline_formatted': format_gestational_age(baseline_value),
                'simulation_mean': sim_mean,
                'simulation_formatted': format_gestational_age(sim_mean),
                'difference': diff,
                'coefficient_of_variation': cv
            }
        else:
            print(f"{p}%\t{format_gestational_age(baseline_value)} ({baseline_value:.2f})\t"
                  f"无模拟数据\t\t-\t\t-")
            
            comparison_results[p] = {
                'baseline_value': baseline_value,
                'baseline_formatted': format_gestational_age(baseline_value),
                'simulation_mean': None,
                'simulation_formatted': None,
                'difference': None,
                'coefficient_of_variation': None
            }
    
    return comparison_results


def visualize_simulation_results(simulation_results, sigma_y, sigma_time, percentiles=[50, 90]):
    """
    可视化模拟结果的分布图，直观查看变异程度
    
    参数:
    simulation_results: 模拟结果
    sigma_y: Y染色体浓度噪声标准差
    sigma_time: 检测时间噪声标准差
    percentiles: 分位数列表
    """
    if 'percentile_times' not in simulation_results:
        print("无法可视化模拟结果分布：缺少详细时间数据")
        return
    
    # 创建图形
    fig, axes = plt.subplots(len(percentiles), 1, figsize=(10, 4*len(percentiles)))
    if len(percentiles) == 1:
        axes = [axes]
    
    for i, p in enumerate(percentiles):
        if p in simulation_results['percentile_times']:
            times = simulation_results['percentile_times'][p]
            
            # 绘制直方图
            axes[i].hist(times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_xlabel('时间 (天)')
            axes[i].set_ylabel('频次')
            axes[i].set_title(f'{p}%分位时间分布 (σ_y={sigma_y}, σ_time={sigma_time})')
            axes[i].grid(True, alpha=0.3)
            
            # 添加均值线
            mean_time = np.mean(times)
            axes[i].axvline(mean_time, color='red', linestyle='--', linewidth=2, 
                           label=f'均值: {format_gestational_age(mean_time)} ({mean_time:.2f}天)')
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()


def sensitivity_analysis(results_dict, percentiles=[50, 90]):
    """
    通过敏感性分析确认哪些参数对输出影响最大
    
    参数:
    results_dict: 不同参数设置下的结果字典
    percentiles: 分位数列表
    """
    # 提取不同参数下的结果
    y_only_results = {}  # 只有Y浓度噪声的结果
    time_only_results = {}  # 只有时间噪声的结果
    combined_results = {}  # 组合噪声的结果
    
    for key, result in results_dict.items():
        if key.startswith('y_') and '_time_' not in key:
            sigma_y = float(key.split('_')[1])
            y_only_results[sigma_y] = result
        elif key.startswith('time_'):
            sigma_time = float(key.split('_')[1])
            time_only_results[sigma_time] = result
        elif '_time_' in key:
            parts = key.split('_')
            sigma_y = float(parts[1])
            sigma_time = float(parts[3])
            combined_results[(sigma_y, sigma_time)] = result
    
    # 分析每个分位数
    for p in percentiles:
        print(f"\n=== {p}%分位数敏感性分析 ===")
        
        # 分析Y浓度噪声的影响
        if y_only_results:
            print("\nY染色体浓度噪声影响:")
            y_sigmas = sorted(y_only_results.keys())
            y_means = []
            for sigma in y_sigmas:
                if p in y_only_results[sigma]['stats']:
                    y_means.append(y_only_results[sigma]['stats'][p]['mean'])
            
            if y_means:
                print("σ_y\t均值\t\t变异")
                print("-" * 30)
                baseline_mean = y_means[0]  # 以最小噪声为基准
                for i, sigma in enumerate(y_sigmas):
                    diff = y_means[i] - baseline_mean
                    print(f"{sigma}\t{format_gestational_age(y_means[i])} ({y_means[i]:.2f})\t{diff:.2f}")
        
        # 分析时间噪声的影响
        if time_only_results:
            print("\n检测时间噪声影响:")
            time_sigmas = sorted(time_only_results.keys())
            time_means = []
            for sigma in time_sigmas:
                if p in time_only_results[sigma]['stats']:
                    time_means.append(time_only_results[sigma]['stats'][p]['mean'])
            
            if time_means:
                print("σ_time\t均值\t\t变异")
                print("-" * 30)
                baseline_mean = time_means[0]  # 以最小噪声为基准
                for i, sigma in enumerate(time_sigmas):
                    diff = time_means[i] - baseline_mean
                    print(f"{sigma}\t{format_gestational_age(time_means[i])} ({time_means[i]:.2f})\t{diff:.2f}")


def save_results_to_json(results, filename=None):
    """
    将结果保存到JSON文件
    
    参数:
    results: 要保存的结果字典
    filename: 文件名，如果为None则自动生成
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monte_carlo_results_{timestamp}.json"
    
    # 确保文件保存在当前脚本所在目录
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    # 处理numpy数组和特殊数据类型，使其可以序列化为JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到文件: {filepath}")
    return filepath


def run_sensitivity_analysis(original_data, sigma_y_values=[0.05, 0.1], sigma_time_values=[2.0, 5.0], 
                            n_simulations=100, check_time='检测孕周', percentiles=[50, 90], 
                            visualize=False):
    """
    运行敏感性分析，测试不同噪声水平下的结果稳定性
    
    参数:
    original_data: 原始数据
    sigma_y_values: Y染色体浓度的不同噪声标准差值
    sigma_time_values: 检测时间的不同噪声标准差值
    n_simulations: 每种情况下的模拟次数
    check_time: 检查时间列名
    percentiles: 要分析的分位数列表
    visualize: 是否显示可视化结果
    
    返回:
    不同sigma值下的分析结果
    """
    results = {}
    
    print("开始敏感性分析...")
    
    # 测试不同的Y浓度噪声水平（时间噪声为0）
    print("\n--- 测试Y染色体浓度噪声对结果的影响 (时间噪声σ_time=0) ---")
    for sigma_y in sigma_y_values:
        print(f"\n分析σ_y={sigma_y}的情况:")
        result = monte_carlo_simulation(original_data, n_simulations, sigma_y, 0, check_time, percentiles)
        if result:
            results[f'y_{sigma_y}'] = result
            # 与基准数据比较
            comparison = compare_with_baseline(original_data, result, percentiles, check_time)
            result['comparison'] = comparison
            
            # 可视化结果
            # if visualize:
            #     visualize_simulation_results(result, sigma_y, 0, percentiles)
    
    # 测试不同的时间噪声水平（Y浓度噪声为0）
    print("\n--- 测试检测时间噪声对结果的影响 (Y浓度噪声σ_y=0) ---")
    for sigma_time in sigma_time_values:
        print(f"\n分析σ_time={sigma_time}的情况:")
        result = monte_carlo_simulation(original_data, n_simulations, 0, sigma_time, check_time, percentiles)
        if result:
            results[f'time_{sigma_time}'] = result
            # 与基准数据比较
            comparison = compare_with_baseline(original_data, result, percentiles, check_time)
            result['comparison'] = comparison
            
            # # 可视化结果
            # if visualize:
            #     visualize_simulation_results(result, 0, sigma_time, percentiles)
    
    # 测试同时存在两种噪声的情况
    print("\n--- 测试同时存在两种噪声的情况 ---")
    for sigma_y in sigma_y_values:
        for sigma_time in sigma_time_values:
            print(f"\n分析σ_y={sigma_y}, σ_time={sigma_time}的情况:")
            result = monte_carlo_simulation(original_data, n_simulations, sigma_y, sigma_time, check_time, percentiles)
            if result:
                results[f'y_{sigma_y}_time_{sigma_time}'] = result
                # 与基准数据比较
                comparison = compare_with_baseline(original_data, result, percentiles, check_time)
                result['comparison'] = comparison
                
                # 可视化结果
                if visualize and sigma_y == sigma_y_values[len(sigma_y_values)-1] and sigma_time == sigma_time_values[len(sigma_time_values)-1]:
                    visualize_simulation_results(result, sigma_y, sigma_time, percentiles)
    
    # 执行敏感性分析
    sensitivity_analysis(results, percentiles)
    
    # 保存结果到JSON文件（只保存比较结果）
    comparison_only_results = {}
    for key, result in results.items():
        if 'comparison' in result:
            comparison_only_results[key] = result['comparison']
    
    save_results_to_json(comparison_only_results)
    
    return results


if __name__ == "__main__":
    # 加载数据
    data = Data.data
    
    # 运行敏感性分析
    sensitivity_results = run_sensitivity_analysis(
        original_data=data, 
        sigma_y_values=[0.05, 0.1], 
        sigma_time_values=[2.0, 5.0],  # 时间噪声使用较大的值，因为时间是以天为单位的
        n_simulations=500,
        percentiles=[90],  # 可以修改为其他分位数，如 [25, 50, 75, 90]
        visualize=True  # 显示可视化结果
    )
    
    print("\n敏感性分析完成。")