# 获取项目根目录路径
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from significance_analysis import LogisticRegressionAnalysis
from data.util.data_manager import Data
import numpy as np

# 初始化模型
model = LogisticRegressionAnalysis()

# 加载数据并创建二分类标签（示例使用中位数阈值）
data = Data.data
X_train = data[['检测孕周', '孕妇BMI']]
y_train = np.where(data['Y染色体浓度'] > data['Y染色体浓度'].median(), 1, 0)

# 训练模型
model.fit(X_train, y_train)

# 获取显著性分析结果
results = model.calculate_significance()

# 打印报告
import pprint
print("\n逻辑回归显著性分析报告:")
pprint.pprint({
    '回归系数': results['coefficients'],
    '标准误差': results['std_error'],
    'Z值': results['z_values'],
    'P值': results['p_values'],
    '95%置信区间下限': results['ci_95%_lower'],
    '95%置信区间上限': results['ci_95%_upper']
}, width=100, depth=2, indent=2)