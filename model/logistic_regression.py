import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.util.data_manager import Data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogisticRegressionModel:
    def __init__(self, C=1.0, random_state=42):
        '''初始化逻辑回归模型
        Args:
            C: 正则化强度的倒数，必须为正数
            random_state: 随机种子
        '''
        self.model = LogisticRegression(C=C, random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X, y):
        '''训练逻辑回归模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
        '''
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X):
        '''返回预测类别
        Args:
            X: 特征矩阵 (n_samples, n_features)
        Returns:
            预测的类别标签
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        '''返回预测概率
        Args:
            X: 特征矩阵 (n_samples, n_features)
        Returns:
            预测各类别的概率
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        '''返回准确率评估分数
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 真实标签 (n_samples,)
        Returns:
            准确率分数
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)

    def get_coefficients(self):
        '''获取模型系数
        Returns:
            模型系数
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.coef_

    def get_intercept(self):
        '''获取模型截距
        Returns:
            模型截距
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        return self.model.intercept_

    def analyze_feature_importance(self, feature_names=None):
        '''分析特征重要性
        Args:
            feature_names: 特征名称列表
        Returns:
            特征重要性分析结果
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        coefficients = self.model.coef_[0]
        importance = np.abs(coefficients)
        
        if feature_names is None:
            feature_names = [f"特征_{i}" for i in range(len(coefficients))]
            
        # 创建特征重要性排序
        feature_importance = list(zip(feature_names, coefficients, importance))
        feature_importance.sort(key=lambda x: x[2], reverse=True)
        
        return feature_importance

    def find_optimal_feature_value(self, feature_index=0, value_range=None, target_class=1):
        '''找到最优特征值（使目标类别概率最大）
        Args:
            feature_index: 特征索引
            value_range: 特征值范围 (min, max)
            target_class: 目标类别
        Returns:
            最优特征值和对应概率
        '''
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        if value_range is None:
            # 使用默认范围
            value_range = (0, 10)  # 默认范围，实际使用时应该根据数据调整
            
        # 生成特征值范围
        feature_values = np.linspace(value_range[0], value_range[1], 1000)
        
        # 计算每个特征值对应的概率
        max_prob = -1
        optimal_value = None
        
        for value in feature_values:
            # 创建样本数据（只设置指定特征，其他特征使用均值）
            sample = np.zeros((1, len(self.scaler.mean_)))
            sample[0, feature_index] = (value - self.scaler.mean_[feature_index]) / self.scaler.scale_[feature_index]
            
            # 预测概率
            prob = self.model.predict_proba(sample)[0][target_class]
            
            if prob > max_prob:
                max_prob = prob
                optimal_value = value
                
        return optimal_value, max_prob

if __name__ == "__main__":
    data = Data.data
    
    # 为了演示逻辑回归，我们需要一个分类目标变量
    # 我们创建一个二分类目标变量：Y染色体浓度是否高于中位数
    y_original = data['Y染色体浓度'].dropna()
    median_value = y_original.median()
    y_binary = (y_original > median_value).astype(int)
    
    # 选择特征
    X = data.loc[y_binary.index, ['检测孕周', '孕妇BMI']].dropna()
    # 确保y_binary与X有相同的索引
    y = y_binary.loc[X.index]
    
    # 创建并训练逻辑回归模型
    lr_model = LogisticRegressionModel(C=1.0)
    lr_model.train(X, y)
    
    # 输出结果
    print("模型系数:", lr_model.get_coefficients())
    print("模型截距:", lr_model.get_intercept())
    print("准确率:", lr_model.evaluate(X, y))
    
    # 特征重要性分析
    feature_importance = lr_model.analyze_feature_importance(['检测孕周', '孕妇BMI'])
    print("\n特征重要性:")
    for name, coef, importance in feature_importance:
        print(f"  {name}: 系数={coef:.4f}, 重要性={importance:.4f}")
    
    # 预测示例
    sample_data = X[:5]
    predictions = lr_model.predict(sample_data)
    probabilities = lr_model.predict_proba(sample_data)
    
    print("\n示例预测:")
    for i in range(len(sample_data)):
        print(f"样本 {i+1}: 特征={sample_data.iloc[i].values}, 预测类别={predictions[i]}, 概率={probabilities[i]}")
    
    # 寻找最优检测孕周
    # 首先获取检测孕周在特征中的索引
    feature_names = ['检测孕周', '孕妇BMI']
    week_index = feature_names.index('检测孕周')
    
    # 获取检测孕周的范围
    week_min, week_max = X['检测孕周'].min(), X['检测孕周'].max()
    
    # 寻找最优孕周
    optimal_week, max_prob = lr_model.find_optimal_feature_value(
        feature_index=week_index, 
        value_range=(week_min, week_max), 
        target_class=1  # 高Y染色体浓度类别
    )
    
    print(f"\n最优检测孕周分析:")
    print(f"  检测孕周范围: {week_min:.2f} - {week_max:.2f}")
    print(f"  最优孕周: {optimal_week:.2f}")
    print(f"  对应高Y染色体浓度概率: {max_prob:.4f}")