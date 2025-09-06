import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from data.util.data_manager import Data
class LinearRegressionAnalysis:
    def __init__(self):
        """初始化线性回归分析模型"""
        self.model = LinearRegression()
        self.coef_ = None      # 回归系数
        self.intercept_ = None  # 截距项
        self.std_err = None    # 系数标准误
        self.t_values = None   # t统计量
        self.p_values = None   # p值

    def fit(self, X, y):
        # 存储训练数据用于后续分析
        self.X = X
        self.y = y
        
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

        # 计算标准误差
        y_pred = self.model.predict(X)
        residuals = y - y_pred
        sigma_sq = np.var(residuals, ddof=X.shape[1]+1)
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        cov_matrix = sigma_sq * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        self.std_err = np.sqrt(np.diag(cov_matrix))[1:]  # 排除截距项

    def calculate_significance(self, alpha=0.05):
        # 使用存储的X数据
        n_samples = self.X.shape[0]
        n_features = len(self.coef_)
        
        # 计算t统计量和p值
        self.t_values = self.coef_ / self.std_err
        self.p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n_samples-n_features-1)) 
                        for t in self.t_values]
        
        # 计算置信区间
        t_critical = stats.t.ppf(1 - alpha/2, n_samples-n_features-1)
        ci_lower = self.coef_ - t_critical * self.std_err
        ci_upper = self.coef_ + t_critical * self.std_err

        return {
            'coefficients': self.coef_,
            'std_error': self.std_err,
            't_values': self.t_values,
            'p_values': self.p_values,
            f'ci_{int((1-alpha)*100)}%_lower': ci_lower,
            f'ci_{int((1-alpha)*100)}%_upper': ci_upper
        }

# 使用示例
if __name__ == "__main__":
    model = LinearRegressionAnalysis()
    data = Data.data
    X_train = data[['检测孕周', '孕妇BMI']]
    y_train = data['Y染色体浓度']
    model.fit(X_train, y_train)
    results = model.calculate_significance()
    print("显著特征:", [i for i,p in enumerate(results['p_values']) if p < 0.05])
        

