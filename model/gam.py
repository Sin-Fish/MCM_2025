import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.util.data_manager import Data
from pygam import LinearGAM, s
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class GAMModel:
    def __init__(self, n_splines=10, lam=0.6):
        '''初始化GAM模型
        Args:
            n_splines: 样条数量
            lam: 正则化参数
        '''
        self.model = None
        self.n_splines = n_splines
        self.lam = lam
        self.is_trained = False

    def train(self, X, y):
        '''训练GAM模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
        '''
        # 创建GAM模型，为每个特征使用平滑样条
        n_features = X.shape[1]
        terms = s(0, n_splines=self.n_splines, lam=self.lam)
        for i in range(1, n_features):
            terms += s(i, n_splines=self.n_splines, lam=self.lam)
            
        self.model = LinearGAM(terms)
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        '''返回预测值
        Args:
            X: 特征矩阵 (n_samples, n_features)
        Returns:
            预测值
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.predict(X)

    def evaluate(self, X, y):
        '''返回R平方评估分数
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 真实标签 (n_samples,)
        Returns:
            R平方分数
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        y_pred = self.model.predict(X)
        return r2_score(y, y_pred)
    
    def evaluate_detailed(self, X, y):
        '''返回详细的评估指标
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 真实标签 (n_samples,)
        Returns:
            包含多个评估指标的字典
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

if __name__ == "__main__":
    data = Data.data
    X = data[['检测孕周', '孕妇BMI']].dropna()
    y = data.loc[X.index, 'Y染色体浓度']
    
    # 创建并训练GAM模型
    gam_model = GAMModel()
    gam_model.train(X, y)
    
    # 输出结果
    print("模型已训练")
    
    # 获取详细评估指标
    metrics = gam_model.evaluate_detailed(X, y)
    print(f"R平方: {metrics['r2_score']:.6f}")
    print(f"均方误差(MSE): {metrics['mse']:.6f}")
    print(f"均方根误差(RMSE): {metrics['rmse']:.6f}")
    print(f"平均绝对误差(MAE): {metrics['mae']:.6f}")
    
    # 预测示例
    sample_data = X[:5]
    predictions = gam_model.predict(sample_data)
    actual_values = y[:5]
    
    print("\n示例预测:")
    for i in range(len(sample_data)):
        print(f"样本 {i+1}: 特征={sample_data.iloc[i].values}, 实际值={actual_values.iloc[i]}, 预测值={predictions[i]:.6f}")