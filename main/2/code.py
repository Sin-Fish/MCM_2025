import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.util.data_manager import Data
from model.k_means import KMeansCluster
from data.util.draw import draw_col_seaborn
from model.logistic_regression import LogisticRegressionModel

if __name__ == "__main__":
    data = Data.data
   
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

    # 分离不同类别的数据
    clustered_data = X.copy()
    clustered_data['cluster'] = kmeans.get_labels()
    
    # 对每类数据进行可视化
    print("\n=== 各簇数据可视化 ===")
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
        print(f"\n簇 {cluster_id} 数据可视化:")
        if len(cluster_data) > 0:
            # 可视化该簇的检测孕周与Y染色体浓度关系
            # 注意：这里需要原始数据，而不是当前特征数据
            # 所以我们需要从原始数据中筛选出属于该簇的样本
            original_indices = cluster_data.index
            original_data_for_cluster = data.loc[original_indices]
            
            # 绘制检测孕周与Y染色体浓度的关系
            if len(original_data_for_cluster) > 0:
                # 创建临时数据用于绘图
                temp_data = original_data_for_cluster.copy()
                temp_data = temp_data[['检测孕周', 'Y染色体浓度']].dropna()
                if len(temp_data) > 0:
                    print(f"簇 {cluster_id} 包含 {len(temp_data)} 个有效样本")
                    
    # 对每类数据进行逻辑回归确定检测孕周为什么时候最佳
    print("\n=== 各簇逻辑回归分析 ===")
    optimal_weeks = []
    
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
        print(f"\n簇 {cluster_id} 逻辑回归分析:")
        
        if len(cluster_data) > 10:  # 确保有足够的样本进行分析
            # 获取该簇在原始数据中的索引
            original_indices = cluster_data.index
            original_data_for_cluster = data.loc[original_indices]
            
            # 准备逻辑回归所需的数据
            analysis_data = original_data_for_cluster[['检测孕周', 'Y染色体浓度']].dropna()
            
            if len(analysis_data) > 10:  # 确保有足够的样本
                # 创建二分类目标变量：Y染色体浓度是否高于该簇的中位数
                y_median = analysis_data['Y染色体浓度'].median()
                y_binary = (analysis_data['Y染色体浓度'] > y_median).astype(int)
                
                # 特征为检测孕周
                X_lr = analysis_data[['检测孕周']]
                y_lr = y_binary
                
                if len(X_lr) == len(y_lr) and len(X_lr) > 0:
                    # 创建并训练逻辑回归模型
                    lr_model = LogisticRegressionModel(C=1.0)
                    lr_model.train(X_lr, y_lr)
                    
                    # 评估模型
                    accuracy = lr_model.evaluate(X_lr, y_lr)
                    print(f"  模型准确率: {accuracy:.4f}")
                    
                    # 获取模型系数
                    coef = lr_model.get_coefficients()
                    intercept = lr_model.get_intercept()
                    print(f"  模型系数: {coef}")
                    print(f"  模型截距: {intercept}")
                    
                    # 分析哪个孕周最佳
                    # 生成孕周范围
                    weeks = np.linspace(analysis_data['检测孕周'].min(), 
                                      analysis_data['检测孕周'].max(), 100)
                    
                    # 计算每个孕周对应的高Y染色体浓度概率
                    prob_high = []
                    for week in weeks:
                        prob = lr_model.predict_proba(np.array([[week]]))
                        prob_high.append(prob[0][1])  # 高浓度的概率
                    
                    # 找到概率最高的孕周
                    best_week_idx = np.argmax(prob_high)
                    best_week = weeks[best_week_idx]
                    best_prob = prob_high[best_week_idx]
                    
                    print(f"  最佳检测孕周: {best_week:.2f}周")
                    print(f"  对应高Y染色体浓度概率: {best_prob:.4f}")
                    
                    optimal_weeks.append((cluster_id, best_week, best_prob))
                else:
                    print("  样本数据不足，无法进行逻辑回归分析")
            else:
                print("  有效数据不足，无法进行逻辑回归分析")
        else:
            print("  样本数量不足，无法进行逻辑回归分析")
    
    # 输出总体最佳孕周
    print("\n=== 总体分析结果 ===")
    if optimal_weeks:
        for cluster_id, week, prob in optimal_weeks:
            print(f"簇 {cluster_id}: 最佳孕周 {week:.2f}天 (高Y染色体浓度概率: {prob:.4f})")
    else:
        print("没有足够的数据进行分析")