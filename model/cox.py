import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.util.data_manager import Data
try:
    from lifelines import CoxPHFitter
    COX_AVAILABLE = True
except ImportError:
    COX_AVAILABLE = False
    print("警告: 未安装 lifelines 库，Cox模型不可用。请使用 'pip install lifelines' 安装。")

class CoxModel:
    def __init__(self, alpha=0, penalizer=0):
        '''初始化Cox比例风险模型
        Args:
            alpha: L1正则化参数，0表示无L1正则化
            penalizer: L2正则化参数，0表示无L2正则化
        '''
        if not COX_AVAILABLE:
            raise ImportError("lifelines 库未安装，无法使用Cox模型")
            
        self.model = CoxPHFitter(alpha=alpha, penalizer=penalizer)
        self.is_trained = False
        self.feature_names = None

    def train(self, X, event_col, duration_col):
        '''训练Cox比例风险模型
        Args:
            X: 特征数据框，包含特征列和生存信息
            event_col: 事件列名（是否发生事件）
            duration_col: 时间列名（生存时间）
        '''
        # 创建用于训练的数据
        self.feature_names = [col for col in X.columns if col not in [event_col, duration_col]]
        self.model.fit(X, duration_col=duration_col, event_col=event_col)
        self.is_trained = True

    def predict(self, X):
        '''预测风险分数
        Args:
            X: 特征数据框 (n_samples, n_features)
        Returns:
            风险分数
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.predict_partial_hazard(X)

    def predict_survival_function(self, X):
        '''预测生存函数
        Args:
            X: 特征数据框 (n_samples, n_features)
        Returns:
            生存函数
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.predict_survival_function(X)

    def get_coefficients(self):
        '''获取模型系数
        Returns:
            模型系数
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.params_

    def get_confidence_intervals(self):
        '''获取系数的置信区间
        Returns:
            系数的置信区间
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.confidence_intervals_

    def get_feature_importance(self):
        '''获取特征重要性（基于系数绝对值）
        Returns:
            特征重要性
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        coefficients = self.model.params_
        importance = np.abs(coefficients)
        
        feature_importance = list(zip(coefficients.index, coefficients.values, importance.values))
        feature_importance.sort(key=lambda x: x[2], reverse=True)
        return feature_importance

    def evaluate(self, X, event_col, duration_col):
        '''评估模型性能（使用C指数）
        Args:
            X: 包含特征和生存信息的数据框
            event_col: 事件列名
            duration_col: 时间列名
        Returns:
            C指数
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.concordance_index_

    def print_summary(self):
        '''打印模型摘要信息'''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.print_summary()

if __name__ == "__main__":
    if COX_AVAILABLE:
        # 示例用法
        print("Cox模型已定义，可以用于生存分析任务")
        print("使用 lifelines 库的 CoxPHFitter 实现")
    else:
        print("请安装lifelines库以使用Cox模型: pip install lifelines")