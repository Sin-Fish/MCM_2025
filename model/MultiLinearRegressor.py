from os import name
from sklearn.linear_model import LinearRegression
import numpy as np
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.util.data_manager import Data
    
class MultiLinearRegressor:
    def __init__(self):
        self.model = LinearRegression()
        self.coef_ = None
        self.intercept_ = None

    def train(self, X, y):
        '''训练模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
        '''
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

    def predict(self, X):
        '''返回预测值'''
        return self.model.predict(X)

    def evaluate(self, X, y):
        '''返回R平方评估分数'''
        return self.model.score(X, y)

if __name__ == "__main__":  
    data = Data.data
    X = data[['检测孕周', '孕妇BMI']]
    y = data['Y染色体浓度']
    regressor = MultiLinearRegressor()
    regressor.train(X, y)
    print("系数:", regressor.coef_)
    print("截距:", regressor.intercept_)
    print("R平方:", regressor.evaluate(X, y))  