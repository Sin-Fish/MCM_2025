import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
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

class LogisticRegressionAnalysis:
    def __init__(self):
        """初始化逻辑回归分析模型"""
        self.class_weight={0:1,1:1}
        self.model = LogisticRegression(class_weight=self.class_weight, random_state=42, max_iter=1000)
        self.coef_ = None       # 回归系数
        self.intercept_ = None  # 截距项
        self.std_err = None     # 系数标准误
        self.z_values = None    # z统计量
        self.p_values = None    # p值
        self.threshold = 0.11
    def fit(self, X, y):
        self.model.fit(X, y)
        self.coef_ = self.model.coef_[0]
        self.intercept_ = self.model.intercept_[0]

        # 计算标准误差（使用Hessian矩阵）
        pred_probs = self.model.predict_proba(X)[:, 1]
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        W = np.diag(pred_probs * (1 - pred_probs))
        hessian = X_with_intercept.T @ W @ X_with_intercept
        cov_matrix = np.linalg.inv(hessian)
        self.std_err = np.sqrt(np.diag(cov_matrix))[1:]  # 排除截距项

    def calculate_significance(self, alpha=0.05):
        # 计算Wald检验统计量
        self.z_values = self.coef_ / self.std_err
        self.p_values = [2 * (1 - stats.norm.cdf(np.abs(z))) for z in self.z_values]

        # 计算置信区间
        z_critical = stats.norm.ppf(1 - alpha/2)
        ci_lower = self.coef_ - z_critical * self.std_err
        ci_upper = self.coef_ + z_critical * self.std_err

        return {
            'coefficients': self.coef_,
            'std_error': self.std_err,
            'z_values': self.z_values,
            'p_values': self.p_values,
            f'ci_{int((1-alpha)*100)}%_lower': ci_lower,
            f'ci_{int((1-alpha)*100)}%_upper': ci_upper
        }
    def predict_proba(self, X):
        """预测每个样本为0和1的概率"""
        return self.model.predict_proba(X)

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.threshold
        """根据阈值预测类别"""
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    def evaluate(self, X, y, threshold=None):
        if threshold is None:
            threshold = self.threshold
        y_pred = self.predict(X, threshold)
        proba = self.predict_proba(X)[:, 1]
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, proba)
        cm = confusion_matrix(y, y_pred)
        return {
            "Accuracy": acc,
            "AUC": auc,
            "ConfusionMatrix": cm
        }

        

