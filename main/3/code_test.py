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

def test_monte_carlo_with_cox_model():
    """使用Cox模型测试蒙特卡洛框架"""
    print("=== 使用Cox模型测试蒙特卡洛框架 ===")
    
    # 加载数据
    data = Data.data
    
    # 创建K-means聚类模型
    kmeans_model = KMeansCluster(n_clusters=4)
    
    # 创建蒙特卡洛框架实例
    mc_framework = MonteCarloFramework(
        model=None,  # Cox模型将在分析过程中使用，不需要在这里指定
        cluster_model=kmeans_model,
        feature_columns=['孕妇BMI', 'Y染色体浓度']
    )
    
    # 运行敏感性分析
    sensitivity_results = mc_framework.run_sensitivity_analysis(
        original_data=data,
        sigma_y_values=[0.05, 0.1],
        sigma_time_values=[2.0, 5.0],
        n_simulations=100,  # 减少模拟次数以加快测试速度
        percentiles=[50, 90],
        visualize=True
    )
    
    print("\n=== 敏感性分析结果 ===")
    # 显示部分结果
    for key, result in list(sensitivity_results.items())[:3]:
        print(f"{key}: {result.get('n_valid_simulations', 'N/A')} 次有效模拟")
    
    return sensitivity_results

def test_original_functionality():
    """测试原始功能以确保向后兼容性"""
    print("\n=== 测试原始功能 ===")
    
    # 加载数据
    data = Data.data
    
    # 使用原始方法进行蒙特卡洛模拟
    from main.util.monte_carlo_test import run_sensitivity_analysis
    
    sensitivity_results = run_sensitivity_analysis(
        original_data=data,
        sigma_y_values=[0.05, 0.1],
        sigma_time_values=[2.0, 5.0],
        n_simulations=50,  # 减少模拟次数以加快测试速度
        percentiles=[50, 90],
        visualize=False
    )
    
    print("原始功能测试完成")
    return sensitivity_results

if __name__ == "__main__":
    print("开始测试蒙特卡洛框架...")
    
    # 测试新框架
    new_results = test_monte_carlo_with_cox_model()
    
    # 测试原始功能
    original_results = test_original_functionality()
    
    print("\n测试完成！")