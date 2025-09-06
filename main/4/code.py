import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from data.util.data_manager import Data
from main.util.significance_analysis import LogisticRegressionAnalysis
from sklearn.preprocessing import StandardScaler
if __name__ == '__main__':
    data = Data.data
    # 初始化模型
    model = LogisticRegressionAnalysis()
    #取特征
    X = data[['年龄',
              'X染色体的Z值',
              '13号染色体的Z值',
              '18号染色体的Z值',
              '21号染色体的Z值',
              'GC含量',
              '原始读段数',
              '在参考基因组上比对的比例',
              '重复读段的比例',
              '唯一比对的读段数',
              '孕妇BMI'
              ]]
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 取目标
    # 将NaN转为0，非空转为1
    y = (~data['染色体的非整倍体'].isna()).astype(int)  # NaN→0，非空→1
    # 训练模型
    model.fit(X,y)
    # 计算显著性
    result = model.calculate_significance()
    print("coefficients:"+str(result['coefficients']))
    print("std_error:"+str(result['std_error']))
    print("z_values:"+str(result['z_values']))
    print("p_values:"+str(result['p_values']))
    print("ci_95%_lower:"+str(result['ci_95%_lower']))
    print("ci_95%_upper:"+str(result['ci_95%_upper']))

    

