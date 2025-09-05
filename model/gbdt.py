from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.util.data_manager import Data
class GBDTRegressor:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        self.model = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth
        )

    def train(self, X, y):
        '''训练梯度提升树模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
        '''
        self.model.fit(X, y)

    def predict(self, X):
        '''返回预测值'''
        return self.model.predict(X)

    def get_feature_importance(self):
        '''获取特征重要性'''
        return self.model.feature_importances_
if __name__ == "__main__":  
    data = Data.data
    X = data[['检测孕周', '孕妇BMI']]
    y = data['Y染色体浓度']
    gbdt = GBDTRegressor(n_estimators=150, max_depth=5)
    gbdt.train(X, y)
    print("训练分数:", gbdt.model.train_score_.mean())
    print("特征重要性:", gbdt.get_feature_importance())
    print("测试预测示例:", gbdt.predict(X[:5]))