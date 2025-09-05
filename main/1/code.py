import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from data.util.data_manager import Data
from model.MultiLinearRegressor import MultiLinearRegressor


if __name__ == "__main__":  
    data = Data.data
    X = data[['更新孕周', '孕妇BMI']]
    y = data['Y染色体浓度']
    regressor = MultiLinearRegressor()
    regressor.train(X, y)
    print("系数:", regressor.coef_)
    print("截距:", regressor.intercept_)
    print("R平方:", regressor.evaluate(X, y))  