import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
import numpy as np
from data.util.data_manager import Data
from main.util.significance_analysis import LogisticRegressionAnalysis
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
if __name__ == '__main__':
    data = Data.data
    # 初始化模型
    model = LogisticRegressionAnalysis()
    #取特征
    raw_X = data[['年龄',
              'X染色体的Z值',
              '13号染色体的Z值',
              '18号染色体的Z值',
              '21号染色体的Z值',
              'GC含量',
              '原始读段数',
              '孕妇BMI'
              ]]
    # 取目标
    # 将NaN转为0，非空转为1
    y = (~data['染色体的非整倍体'].isna()).astype(int)  # NaN→0，非空→1
     # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        raw_X, y, test_size=0.3, random_state=42, stratify=y
    )
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 把标准化后的数据转为 DataFrame
    X_df = pd.DataFrame(X_train, columns=raw_X.columns)
    # 计算 VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

    print("VIF 检查结果：")
    print(vif_data)
    # 训练模型
    model.fit(X_train,y_train)
    # 计算显著性
    result = model.calculate_significance()
    summary_df = pd.DataFrame({
    "Feature": raw_X.columns,
    "Coefficient": result['coefficients'],
    "StdErr": result['std_error'],
    "z_value": result['z_values'],
    "p_value": result['p_values'],
    "CI_lower": result['ci_95%_lower'],
    "CI_upper": result['ci_95%_upper']
})

    # 在测试集上预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 计算 FPR, TPR 和 阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # 计算 Youden’s J statistic 找最优阈值
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
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    print("最优阈值:", optimal_threshold)
    print("对应 TPR:", tpr[idx], "FPR:", fpr[idx])

    print("\n模型评估：")
    print("准确率 (Accuracy):", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

    

