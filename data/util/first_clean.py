import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.util.data_manager import Data

data = Data.data
#策略函数：
class Strategy:
    def __init__(self,file_name = "first_y.csv"):
        self.file_name = file_name
    

    def avg_data(self,):
         #获取每个孕妇的第一个y浓度达标检测和最后一个不达标检测的时点
         
         # Y染色体浓度阈值
         Y_THRESHOLD = 0.04
         
         # 按孕妇代码分组
         grouped = data.groupby('孕妇代码')
         
         results = []
         
         for woman_id, group in grouped:
             # 按检测孕周排序
             group = group.sort_values('检测孕周')
             
             # 找到所有达标检测（Y染色体浓度 >= 0.04）
             qualified_tests = group[group['Y染色体浓度'] >= Y_THRESHOLD]
             
             # 找到所有不达标检测（Y染色体浓度 < 0.04）
             unqualified_tests = group[group['Y染色体浓度'] < Y_THRESHOLD]
             
             # 特殊处理（没有达标检测数据的孕妇不保存，只有达标检测的数据直接以该次检测时点为最早时点）
             if len(qualified_tests) == 0:
                 continue
             
             # 第一个达标时点
             if len(qualified_tests) > 0:
                 first_qualified_time = qualified_tests.iloc[0]['检测孕周']
             
             # 最后一个不达标时点
             if len(unqualified_tests) > 0:
                 last_unqualified_time = unqualified_tests.iloc[-1]['检测孕周']
             else:
                 # 只有达标检测的数据直接以该次检测时点为最早时点
                 results.append({
                     '序号': group.iloc[0]['序号'],
                     '孕妇代码': woman_id,
                     '第一次达标时间': first_qualified_time,
                     '检测次数': len(group),
                     'BMI': group.iloc[0]['孕妇BMI']
                 })
                 continue
             
             # 根据第一达标时点和最后不达标时点进行平均数，预测第一次达标的时点
             avg_time = (first_qualified_time + last_unqualified_time) / 2
             
             # 保存数据，格式：序号 孕妇代码 第一次达标时间 检测次数 BMI
             results.append({
                 '序号': group.iloc[0]['序号'],
                 '孕妇代码': woman_id,
                 '第一次达标时间': avg_time,
                 '检测次数': len(group),
                 'BMI': group.iloc[0]['孕妇BMI']
             })
         
         # 保存数据
         result_df = pd.DataFrame(results)
         output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", self.file_name)
         result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    def linear_interpolation_data(self):
        #获取每个孕妇的第一个y浓度达标检测和最后一个不达标检测的时点
         
         # Y染色体浓度阈值
         Y_THRESHOLD = 0.04
         
         # 按孕妇代码分组
         grouped = data.groupby('孕妇代码')
         
         results = []
         
         for woman_id, group in grouped:
             # 按检测孕周排序
             group = group.sort_values('检测孕周')
             
             # 找到所有达标检测（Y染色体浓度 >= 0.04）
             qualified_tests = group[group['Y染色体浓度'] >= Y_THRESHOLD]
             
             # 找到所有不达标检测（Y染色体浓度 < 0.04）
             unqualified_tests = group[group['Y染色体浓度'] < Y_THRESHOLD]
             
             # 特殊处理（没有达标检测数据的孕妇不保存，只有达标检测的数据直接以该次检测时点为最早时点）
             if len(qualified_tests) == 0:
                 continue
             
             # 第一个达标时点和浓度
             if len(qualified_tests) > 0:
                 first_qualified_time = qualified_tests.iloc[0]['检测孕周']
                 first_qualified_concentration = qualified_tests.iloc[0]['Y染色体浓度']
             
             # 最后一个不达标时点和浓度
             if len(unqualified_tests) > 0:
                 last_unqualified_time = unqualified_tests.iloc[-1]['检测孕周']
                 last_unqualified_concentration = unqualified_tests.iloc[-1]['Y染色体浓度']
             else:
                 # 只有达标检测的数据直接以该次检测时点为最早时点
                 results.append({
                     '序号': group.iloc[0]['序号'],
                     '孕妇代码': woman_id,
                     '第一次达标时间': first_qualified_time,
                     '检测次数': len(group),
                     'BMI': group.iloc[0]['孕妇BMI']
                 })
                 continue
             
             # 根据第一达标时点和最后不达标时点进行线性插值，预测第一次达标的时点
             # 线性插值公式：x = x1 + (y-y1) * (x2-x1) / (y2-y1)
             # 其中 y 是阈值 0.04
             if first_qualified_concentration != last_unqualified_concentration:
                 interpolated_time = last_unqualified_time + (Y_THRESHOLD - last_unqualified_concentration) * \
                                    (first_qualified_time - last_unqualified_time) / \
                                    (first_qualified_concentration - last_unqualified_concentration)
             else:
                 interpolated_time = (first_qualified_time + last_unqualified_time) / 2
             
             # 保存数据
             results.append({
                 '序号': group.iloc[0]['序号'],
                 '孕妇代码': woman_id,
                 '第一次达标时间': interpolated_time,
                 '检测次数': len(group),
                 'BMI': group.iloc[0]['孕妇BMI']
             })
         
         # 保存数据
         result_df = pd.DataFrame(results)
         output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", self.file_name)
         result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
      
        
if __name__ == "__main__":
    strategy = Strategy("first_y_3.csv")
    strategy.avg_data()
    #strategy.linear_interpolation_data()