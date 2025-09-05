import pandas as pd
import openpyxl
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

from data_manager import Data

# 设置中文字体支持
def set_chinese_font():
    """
    设置中文字体支持
    """
    # 获取系统可用字体
    font_names = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'FangSong']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找可用的中文字体
    for font_name in font_names:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    else:
        # 如果找不到指定字体，使用系统默认字体
        chinese_fonts = [f for f in available_fonts if any(chinese_char in f for chinese_char in ['Sim', 'Kai', 'Fang', 'Microsoft', 'ST'])]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 在模块加载时设置中文字体
set_chinese_font()

def draw_col_hot_map_seaborn(x_col, y_col,z_col):
    """
    绘制热力图
    param:
    x_col: x轴列号(数字索引)或列名(字符串)
    y_col: y轴列号(数字索引)或列名(字符串)
    """
    try:
        # 获取数据
        if isinstance(x_col, str):
            x_data = data[x_col]
            x_label = x_col
        else:
            x_data = data.iloc[:, x_col]
            x_label = f'列 {x_col}'
            
        if isinstance(y_col, str):
            y_data = data[y_col]
            y_label = y_col
        else:
            y_data = data.iloc[:, y_col]
            y_label = f'列 {y_col}'
        
        # 创建热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(pd.DataFrame([x_data, y_data]).T, annot=True, cmap='coolwarm', fmt='.2f')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'热力图: {x_label} vs {y_label}')
        plt.show()
        
    except Exception as e:
        print(f"绘制热力图时发生错误: {e}")

def draw_col_3d_matplotlib(x_col, y_col, z_col):
    """
    绘制3D散点图
    param:
    x_col: x轴列号(数字索引)或列名(字符串)
    y_col: y轴列号(数字索引)或列名(字符串)
    z_col: z轴列号(数字索引)或列名(字符串)
    """
    try:
        # 获取数据
        if isinstance(x_col, str):
            x_data = data[x_col]
            x_label = x_col
        else:
            x_data = data.iloc[:, x_col]
            x_label = f'列 {x_col}'
            
        if isinstance(y_col, str):
            y_data = data[y_col]
            y_label = y_col
        else:
            y_data = data.iloc[:, y_col]
            y_label = f'列 {y_col}'
            
        if isinstance(z_col, str):
            z_data = data[z_col]
            z_label = z_col
        else:
            z_data = data.iloc[:, z_col]
            z_label = f'列 {z_col}'
        
        # 创建3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D散点图
        ax.scatter(x_data, y_data, z_data, c='blue', marker='o')
        
        # 设置坐标轴标签
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        
        # 设置标题
        ax.set_title(f'3D散点图: {x_label} vs {y_label} vs {z_label}')
        
        # 显示图形
        plt.show()
        
    except Exception as e:
        print(f"绘制3D散点图时发生错误: {e}")

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
            x_label = f'列 {x_col}'
            
        
        if isinstance(y_col, str):
            y_data = data[y_col]
            y_label = y_col
        else:
            y_data = data.iloc[:, y_col]
            y_label = f'列 {y_col}'
        
       
        mask = y_data
        x_data_filtered = x_data[mask]
        y_data_filtered = y_data[mask]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_data_filtered, y=y_data_filtered)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'散点图: {x_label} vs {y_label}')
        plt.show()
    except Exception as e:
        print(f"绘图时发生错误: {e}")
    
if __name__ == "__main__":    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    config = {
        "file_path": os.path.join(project_dir, "data", "cleaned_data.csv"),  
        "save_path": os.path.join(project_dir, "data"),  
    }
    data = Data.data
    #draw_col_seaborn('孕妇BMI', 'Y染色体浓度')

    draw_col_seaborn('检测孕周', 'Y染色体浓度')
    #draw_col_seaborn("生产次数", "胎儿是否健康")

    #draw_col_seaborn("生产次数", "Y染色体浓度")

    #draw_col_3d_matplotlib("孕妇BMI", "21号染色体的Z值", "Y染色体浓度")

    draw_col_3d_matplotlib("孕妇BMI", "检测孕周", "Y染色体浓度")
    #draw_col_hot_map_seaborn("孕妇BMI", "Y染色体浓度")
    # for i in range(data.shape[1]):
        
    #     draw_col_seaborn(data.columns[i], "Y染色体浓度")