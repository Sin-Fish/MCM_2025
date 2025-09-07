import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
import numpy as np
from data.util.data_manager import Data
# 使用GBDT分类模型替换逻辑回归分析模型
from model.gbdt import GBDTClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def create_target_variable(data, target_type):
    """创建目标变量
    Args:
        data: 原始数据
        target_type: 目标类型 ('T13', 'T18', 'T21')
    Returns:
        目标变量Series
    """
    # 检查染色体非整倍体列中是否包含指定类型
    if target_type in ['T13', 'T18', 'T21']:
        y = data['染色体的非整倍体'].str.contains(target_type, na=False).astype(int)
    else:
        y = (~data['染色体的非整倍体'].isna()).astype(int)
    return y

# 创建一个字典来存储所有模型的结果，用于统一输出
all_model_results = {}

if __name__ == '__main__':
    data = Data.data
    
    # 特征选择，添加新的特征
    raw_X = data[['年龄',
              'X染色体的Z值',
              '13号染色体的Z值',
              '18号染色体的Z值',
              '21号染色体的Z值',
              'GC含量',
              '孕妇BMI',
              '重复读段的比例',
              '被过滤掉读段数的比例'
              ]]
    
    # 三种目标类型分别训练模型
    target_types = ['T13', 'T18', 'T21']
    models = {}
    results = {}
    
    # 简单检查Z值与异常的关系
    for target_type in ['T13', 'T18', 'T21']:
        z_col = f'{target_type[1:]}号染色体的Z值'  # 如"13号染色体的Z值"
        z_values = data[z_col]
        is_abnormal = create_target_variable(data, target_type)
        
        print(f"\n{target_type}:")
        print(f"异常组的Z值均值: {z_values[is_abnormal == 1].mean()}")
        print(f"正常组的Z值均值: {z_values[is_abnormal == 0].mean()}")
        print(f"Z值与异常的相关性: {z_values.corr(is_abnormal)}")
    
    for target_type in target_types:
        print(f"\n=== 训练{target_type}预测模型 ===")
        # 创建目标变量
        y = create_target_variable(data, target_type)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            raw_X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 注意：GBDT模型不需要特征标准化，但为了保持一致性，我们仍保留此步骤
        # 但在实际使用中，GBDT可以直接使用原始特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 把标准化后的数据转为 DataFrame
        X_df = pd.DataFrame(X_train_scaled, columns=raw_X.columns)
        # 计算 VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_df.columns
        vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

        print(f"{target_type} VIF 检查结果：")
        print(vif_data)
        
        # 训练GBDT分类模型
        model = GBDTClassifier(n_estimators=150, max_depth=5)
        model.train(X_train_scaled, y_train)
        
        # 获取特征重要性（GBDT模型特有的方法）
        feature_importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            "Feature": raw_X.columns,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)
        
        print(f"\n{target_type} 特征重要性分析结果：")
        print(importance_df)

        # 在测试集上预测
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # 计算 FPR, TPR 和 阈值
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        # 计算 Youden's J statistic 找最优阈值
        J = tpr - fpr
        idx = J.argmax()
        optimal_threshold = thresholds[idx]

        # 绘制 ROC 曲线
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
        plt.scatter(fpr[idx], tpr[idx], color='red', s=100, label=f'Optimal Threshold = {optimal_threshold:.3f}')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'{target_type} ROC Curve with Optimal Threshold')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"{target_type} 最优阈值:", optimal_threshold)
        print(f"{target_type} 对应 TPR:", tpr[idx], "FPR:", fpr[idx])

        print(f"\n{target_type} 模型评估：")
        print("准确率 (Accuracy):", accuracy)
        print("AUC:", auc)
        print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
        
        # 保存模型和结果
        models[target_type] = model
        results[target_type] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'optimal_threshold': optimal_threshold,
            'fpr': fpr,
            'tpr': tpr,
            'importance': importance_df
        }
        
        # 保存结果用于统一输出
        all_model_results[target_type] = {
            'accuracy': accuracy,
            'auc': auc,
            'top3_features': importance_df.head(3)
        }
    
    # 统一输出所有模型的准确率和AUC值，以及前三个重要特征
    print("\n=== 统一输出各模型评估结果 ===")
    print("(特征按重要性从高到低排序)")
    for target_type in target_types:
        result = all_model_results[target_type]
        print(f"\n{target_type}模型:")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  AUC值: {result['auc']:.4f}")
        print("  前三个重要特征 (按重要性从高到低):")
        for i, (index, row) in enumerate(result['top3_features'].iterrows(), 1):
            print(f"    {i}. {row['Feature']} (重要性: {row['Importance']:.4f})")