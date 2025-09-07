import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.util.data_manager import Data
from pygam import LinearGAM, s
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats


class GAMModel:
    def __init__(self, splines=None):
        '''
        广义相加模型回归器
        Args:
            splines: 样条函数配置，默认为None时会自动配置
        '''
        self.splines = splines
        self.model = None
        self.feature_names_ = None
        
    def train(self, X, y):
        '''
        训练GAM模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
        '''
        # 如果X是pandas DataFrame，记录特征名
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f'x{i}' for i in range(X.shape[1])]
            X_array = X
            
        # 如果没有指定样条函数，自动为每个特征配置光滑样条
        if self.splines is None:
            n_features = X_array.shape[1]
            if n_features == 1:
                self.splines = s(0)
            else:
                self.splines = s(0)
                for i in range(1, n_features):
                    self.splines += s(i)
        
        self.model = LinearGAM(self.splines)
        self.model.fit(X_array, y)
    
    def predict(self, X):
        '''
        返回预测值
        Args:
            X: 特征矩阵
        Returns:
            预测值数组
        '''
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        return self.model.predict(X_array)
    
    def evaluate(self, X, y):
        '''
        返回R平方评估分数
        Args:
            X: 特征矩阵
            y: 真实值
        Returns:
            R平方分数
        '''
        from sklearn.metrics import r2_score
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def evaluate_detailed(self, X, y):
        '''
        返回详细的评估指标
        Args:
            X: 特征矩阵
            y: 真实值
        Returns:
            dict: 包含多种评估指标的字典
        '''
        predictions = self.predict(X)
        
        # 计算各种评估指标
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        
        return {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    
    def get_aic(self):
        '''
        获取AIC值
        '''
        if self.model is not None:
            return self.model.statistics_['AIC']
        return None
    
    def get_feature_importance(self):
        '''
        获取特征重要性（通过偏导数的平均绝对值估算）
        '''
        if self.model is not None and self.feature_names_ is not None:
            try:
                # 计算每个特征的平均偏导数绝对值作为重要性指标
                importance = {}
                for i, feature_name in enumerate(self.feature_names_):
                    # 获取该特征的偏导数
                    partial_dependence = self.model.partial_dependence(term=i)
                    avg_abs_derivative = np.mean(np.abs(np.gradient(partial_dependence)))
                    importance[feature_name] = avg_abs_derivative
                
                # 按重要性排序
                importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                return importance
            except:
                return None
        return None
    
    def get_significance_test(self, X_train, y_train):
        '''
        GAM模型的显著性检验
        Args:
            X_train: 训练特征矩阵
            y_train: 训练目标变量
        Returns:
            dict: 包含显著性检验结果
        '''
        if self.model is None:
            return None
            
        try:
            # 获取模型统计信息
            statistics = self.model.statistics_
            
            # 计算预测值和残差
            y_pred = self.predict(X_train)
            residuals = y_train - y_pred
            
            # 计算R²
            r_squared = self.evaluate(X_train, y_train)
            
            # 获取每个样条项的显著性
            smooth_terms = []
            if hasattr(self.model, 'terms'):
                for i, term in enumerate(self.model.terms):
                    if hasattr(term, 'info'):
                        try:
                            # 获取样条项的F统计量和p值
                            term_stats = {
                                'feature': self.feature_names_[i] if i < len(self.feature_names_) else f'Term_{i}',
                                'effective_df': getattr(term, 'edof', None),
                                'chi2': getattr(term, 'chi2', None),
                                'p_value': getattr(term, 'p_value', None)
                            }
                            smooth_terms.append(term_stats)
                        except:
                            smooth_terms.append({
                                'feature': self.feature_names_[i] if i < len(self.feature_names_) else f'Term_{i}',
                                'effective_df': None,
                                'chi2': None,
                                'p_value': None
                            })
            
            # 计算整体模型的F检验
            n = len(y_train)
            effective_df = statistics.get('edof', n - 1)  # 有效自由度
            
            # 使用偏差解释计算F统计量
            y_mean = np.mean(y_train)
            tss = np.sum((y_train - y_mean)**2)  # 总平方和
            rss = np.sum(residuals**2)  # 残差平方和
            
            if effective_df < n and rss > 0:
                f_statistic = ((tss - rss) / effective_df) / (rss / (n - effective_df))
                f_p_value = 1 - stats.f.cdf(f_statistic, effective_df, n - effective_df)
            else:
                f_statistic = None
                f_p_value = None
            
            return {
                'model_significance': {
                    'r_squared': r_squared,
                    'deviance_explained': statistics.get('dev_expl', None),
                    'aic': statistics.get('AIC', None),
                    'bic': statistics.get('BIC', None),
                    'gcv': statistics.get('GCV', None),  # 广义交叉验证
                    'f_statistic': f_statistic,
                    'f_p_value': f_p_value,
                    'effective_df': effective_df
                },
                'smooth_terms': smooth_terms,
                'residual_analysis': {
                    'residuals': residuals,
                    'rss': rss,  # 残差平方和
                    'mse': rss / n
                },
                'model_info': {
                    'n_features': len(self.feature_names_) if self.feature_names_ else 0,
                    'n_samples': n,
                    'splines_config': str(self.splines)
                }
            }
            
        except Exception as e:
            return {
                'error': f'GAM显著性检验出错: {str(e)}',
                'basic_stats': {
                    'r_squared': self.evaluate(X_train, y_train),
                    'aic': self.get_aic()
                }
            }
        return None


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