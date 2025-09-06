# 获取项目根目录路径
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # 根据实际文件位置调整.parent次数
sys.path.insert(0, str(ROOT_DIR))
print(ROOT_DIR)
from significance_analysis import LinearRegressionAnalysis
from data.util.data_manager import Data
model = LinearRegressionAnalysis()
data = Data.data
X_train = data[['检测孕周', '孕妇BMI']]
y_train = data['Y染色体浓度']
model.fit(X_train, y_train)
results = model.calculate_significance()
import pprint

# 打印完整统计结果
print("\n显著性分析报告:")
pprint.pprint({
    '回归系数': results['coefficients'],
    '标准误差': results['std_error'],
    'T值': results['t_values'],
    'P值': results['p_values'],
    '95%置信区间下限': results['ci_95%_lower'],
    '95%置信区间上限': results['ci_95%_upper']
}, width=100, depth=2, indent=2)

# 输出显著特征
print("\n显著特征(p<0.05):", [i for i,p in enumerate(results['p_values']) if p < 0.05])
