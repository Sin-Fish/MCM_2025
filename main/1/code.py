import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from data.util.data_manager import Data
from model.gam import GAMModel
from sklearn.model_selection import train_test_split


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