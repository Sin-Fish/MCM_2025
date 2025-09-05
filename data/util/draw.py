import pandas as pd
import openpyxl
import os
import matplotlib.pyplot as plt
import seaborn as sns

from data_manager import Data


def draw_col_seaborn(x_col, y_col):
    """
    绘制两列数据的散点图
    param:
    x_col: x轴列号(数字索引)或列名(字符串)
    y_col: y轴列号(数字索引)或列名(字符串)
    """
    try:
        if isinstance(x_col, str):
            # 如果是字符串，查找对应的列名
            x_data = data[x_col]
            x_label = x_col
        else:
            # 如果是数字，直接使用列索引
            x_data = data.iloc[:, x_col]
            x_label = f'Column {x_col}'
            
        
        if isinstance(y_col, str):
            
            y_data = data[y_col]
            y_label = y_col
        else:
            
            y_data = data.iloc[:, y_col]
            y_label = f'Column {y_col}'
        
       
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_data, y=y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Scatter plot: {x_label} vs {y_label}')
        plt.show()
    except Exception as e:
        print(f"绘图时发生错误: {e}")
    
if __name__ == "__main__":    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    config = {
        "file_path": os.path.join(project_dir, "data", "cleaned_data.xlsx"),  
        "save_path": os.path.join(project_dir, "data"),  
    }
    data = Data.data
    draw_col_seaborn('孕妇BMI', 'Y染色体浓度')