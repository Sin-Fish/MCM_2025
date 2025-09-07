import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from data.util.data_manager import Data
from model.gam import GAMModel
from sklearn.model_selection import train_test_split
from main.util.significance_analysis import GAMRegressionAnalysis


if __name__ == "__main__":  
    data = Data.data
    X = data[['检测孕周', '孕妇BMI','检测抽血次数','年龄']].dropna()
    y = data.loc[X.index, 'Y染色体浓度']
    
    # 划分训练集和测试集 (80%训练, 20%测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    
    # 训练模型
    regressor = GAMModel()
    regressor.train(X_train, y_train)
    
    # 评估训练集性能
    train_metrics = regressor.evaluate_detailed(X_train, y_train)
    print(f"\n训练集性能:")
    print(f"  R平方: {train_metrics['r2_score']:.6f}")
    print(f"  均方根误差(RMSE): {train_metrics['rmse']:.6f}")
    print(f"  平均绝对误差(MAE): {train_metrics['mae']:.6f}")
    
    # 评估测试集性能
    test_metrics = regressor.evaluate_detailed(X_test, y_test)
    print(f"\n测试集性能:")
    print(f"  R平方: {test_metrics['r2_score']:.6f}")
    print(f"  均方根误差(RMSE): {test_metrics['rmse']:.6f}")
    print(f"  平均绝对误差(MAE): {test_metrics['mae']:.6f}")
    
    # 使用显著性分析工具进行GAM模型显著性分析
    print(f"\nGAM模型显著性分析:")
    gam_analysis = GAMRegressionAnalysis()
    gam_analysis.fit(X_train, y_train)
    significance_result = gam_analysis.calculate_significance()
    
    # 显示每个特征的显著性分析结果
    print(f"\n特征显著性分析结果:")
    print(f"{'特征':<15} {'系数':<12} {'标准误':<12} {'t值':<12} {'p值':<12} {'显著性(α=0.05)':<15}")
    print("-" * 90)
    for feature_result in significance_result['feature_significance']:
        significant = "是" if feature_result['significant'] else "否"
        print(f"{feature_result['feature']:<15} {feature_result['coefficient']:<12.6f} {feature_result['std_error']:<12.6f} "
              f"{feature_result['t_value']:<12.6f} {feature_result['p_value']:<12.6f} {significant:<15}")
    
    # 显示模型统计信息
    print(f"\n模型统计信息:")
    model_stats = significance_result['model_statistics']
    for key, value in model_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")