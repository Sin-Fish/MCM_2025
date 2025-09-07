import os
import sys
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.util.data_manager import data_manager
from model.gam import GAMModel
from sklearn.model_selection import train_test_split

def load_female_data():
    """加载女性胎儿数据"""
    # 修改配置以加载女性数据
    female_file_name = "cleaned_data_female"
    csv_file_path = os.path.join(project_root, "data", f"{female_file_name}.csv")
    xlsx_file_path = os.path.join(project_root, "data", f"{female_file_name}.xlsx")
    
    # 根据实际存在的文件设置路径
    if os.path.exists(csv_file_path):
        file_path = csv_file_path
    elif os.path.exists(xlsx_file_path):
        file_path = xlsx_file_path
    else:
        # 如果都不存在，使用默认的xlsx文件路径
        file_path = xlsx_file_path
    
    config = {
        "file_path": file_path,  
    }
    
    female_data_manager = data_manager(config)
    return female_data_manager.data

if __name__ == "__main__":
    # 加载女性胎儿数据
    female_data = load_female_data()
    print(f"女性胎儿数据加载完成，共有 {len(female_data)} 行数据")
    print("女性胎儿数据列名:")
    print(female_data.columns.tolist())
    
    # 加载原始数据用于训练GAM模型
    from data.util.data_manager import Data
    original_data = Data.data
    
    # 准备训练数据 (使用与1/code.py相同的特征)
    X = original_data[['检测孕周', '孕妇BMI', '检测抽血次数', '年龄', "Y染色体的Z值"]].dropna()
    y = original_data.loc[X.index, 'Y染色体浓度']
    
    print(f"\n原始训练数据:")
    print(f"特征数据形状: {X.shape}")
    print(f"目标数据形状: {y.shape}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练GAM模型 (与1/code.py相同)
    print(f"\n训练GAM模型...")
    regressor = GAMModel()
    regressor.train(X_train, y_train)
    
    # 评估模型性能
    train_metrics = regressor.evaluate_detailed(X_train, y_train)
    test_metrics = regressor.evaluate_detailed(X_test, y_test)
    
    print(f"训练集性能:")
    print(f"  R平方: {train_metrics['r2_score']:.6f}")
    print(f"  均方根误差(RMSE): {train_metrics['rmse']:.6f}")
    print(f"  平均绝对误差(MAE): {train_metrics['mae']:.6f}")
    
    print(f"测试集性能:")
    print(f"  R平方: {test_metrics['r2_score']:.6f}")
    print(f"  均方根误差(RMSE): {test_metrics['rmse']:.6f}")
    print(f"  平均绝对误差(MAE): {test_metrics['mae']:.6f}")
    
    # 准备女性胎儿数据用于预测
    print(f"\n准备女性胎儿数据用于预测...")
    # 检查女性数据中是否包含所需特征列
    required_columns = ['检测孕周', '孕妇BMI', '检测抽血次数', '年龄', 'Y染色体的Z值']
    missing_columns = [col for col in required_columns if col not in female_data.columns]
    
    if missing_columns:
        print(f"警告: 女性胎儿数据中缺少以下列: {missing_columns}")
        available_columns = [col for col in required_columns if col in female_data.columns]
        print(f"仅使用以下可用列: {available_columns}")
        
        # 如果缺少特征，使用训练数据的均值来填充缺失的特征列
        X_female = female_data[available_columns].copy()
        
        # 对于缺失的特征列，使用训练数据中的均值进行填充
        for col in missing_columns:
            if col in X_train.columns:
                mean_value = X_train[col].mean()
                X_female[col] = mean_value
                print(f"使用训练数据中 '{col}' 的均值 {mean_value:.4f} 填充缺失列")
        
        # 重新排列列的顺序，确保与训练数据一致
        X_female = X_female[required_columns]
    else:
        print("所有特征列均在女性胎儿数据中找到")
        # 选择女性数据中的特征列
        X_female = female_data[required_columns].copy()
    
    # 删除包含NaN的行
    X_female = X_female.dropna()
    
    print(f"女性胎儿数据形状: {X_female.shape}")
    
    if len(X_female) > 0:
        # 使用训练好的GAM模型预测女性胎儿的Y染色体浓度
        print(f"\n使用GAM模型预测女性胎儿的Y染色体浓度...")
        y_female_pred = regressor.predict(X_female)
        
        # 将预测结果添加到女性胎儿数据中
        female_data_with_predictions = X_female.copy()
        female_data_with_predictions['预测Y染色体浓度'] = y_female_pred
        
        print(f"预测完成，共预测 {len(y_female_pred)} 行数据")
        print("前5行预测结果:")
        print(female_data_with_predictions[['预测Y染色体浓度'] + required_columns].head())
        
        # 保存结果到文件
        output_file = os.path.join(project_root, "result", "female_y_predictions.csv")
        female_data_with_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存到: {output_file}")
    else:
        print("没有足够的数据进行预测")