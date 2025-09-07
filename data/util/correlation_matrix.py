import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.util.data_manager import Data


def set_chinese_font():
    """
    设置中文字体支持
    """
   
    font_names = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'FangSong']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    
    for font_name in font_names:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    else:
       
        chinese_fonts = [f for f in available_fonts if any(chinese_char in f for chinese_char in ['Sim', 'Kai', 'Fang', 'Microsoft', 'ST'])]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
    
    plt.rcParams['axes.unicode_minus'] = False  


set_chinese_font()


def plot_correlation_matrix(data, columns=None, method='pearson', figsize=(12, 10), 
                           annot=True, cmap='coolwarm', title='特征相关性矩阵'):
    """
    绘制数据列间的相关性矩阵热力图
    
    参数:
    data: pandas DataFrame, 数据集
    columns: list, 需要分析的列名列表，默认为None表示使用所有数值列
    method: str, 计算相关性的方法 ('pearson', 'kendall', 'spearman')
    figsize: tuple, 图形大小
    annot: bool, 是否在格子中显示数值
    cmap: str, 热力图颜色映射
    title: str, 图形标题
    
    返回:
    matplotlib.figure.Figure: 生成的图形对象
    """
    # 如果未指定列，则选择所有数值列
    if columns is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        numeric_data = data[columns]
    
    # 计算相关性矩阵
    corr_matrix = numeric_data.corr(method=method)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 创建热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角遮罩
    sns.heatmap(corr_matrix, 
                mask=mask,  # 应用遮罩只显示下三角
                annot=annot, 
                cmap=cmap, 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()


def plot_feature_correlation(data, target_column, columns=None, method='pearson', 
                            figsize=(10, 8), title='特征与目标变量相关性'):
    """
    绘制特征与目标变量的相关性
    
    参数:
    data: pandas DataFrame, 数据集
    target_column: str, 目标变量列名
    columns: list, 需要分析的特征列名列表，默认为None表示使用所有数值列（除了目标变量）
    method: str, 计算相关性的方法 ('pearson', 'kendall', 'spearman')
    figsize: tuple, 图形大小
    title: str, 图形标题
    
    返回:
    matplotlib.figure.Figure: 生成的图形对象
    """
    # 如果未指定列，则选择所有数值列（除了目标变量）
    if columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in numeric_columns if col != target_column]
    
    # 计算与目标变量的相关性
    correlations = {}
    target_data = data[target_column]
    
    for col in columns:
        if col != target_column:
            corr = data[col].corr(target_data, method=method)
            correlations[col] = corr
    
    # 转换为Series并排序
    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制水平条形图
    colors = ['red' if x < 0 else 'blue' for x in corr_series.values]
    bars = plt.barh(range(len(corr_series)), corr_series.values, color=colors, alpha=0.7)
    
    # 设置标签和标题
    plt.yticks(range(len(corr_series)), corr_series.index)
    plt.xlabel(f'{method.capitalize()} 相关性系数')
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, corr_series.values)):
        plt.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.2f}', 
                va='center', ha='left' if value >= 0 else 'right')
    
    plt.tight_layout()
    
    return plt.gcf()


def print_high_correlation_pairs(data, columns=None, method='pearson', threshold=0.7):
    """
    打印高相关性特征对
    
    参数:
    data: pandas DataFrame, 数据集
    columns: list, 需要分析的列名列表，默认为None表示使用所有数值列
    method: str, 计算相关性的方法 ('pearson', 'kendall', 'spearman')
    threshold: float, 高相关性阈值
    
    返回:
    list: 高相关性特征对列表
    """
    # 如果未指定列，则选择所有数值列
    if columns is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        numeric_data = data[columns]
    
    # 计算相关性矩阵
    corr_matrix = numeric_data.corr(method=method)
    
    # 找到高相关性对
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                pair = {
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                }
                high_corr_pairs.append(pair)
    
    # 按相关性绝对值排序
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # 打印结果
    print(f"高相关性特征对 (阈值: {threshold}, 方法: {method}):")
    print("-" * 50)
    for pair in high_corr_pairs:
        print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
    
    return high_corr_pairs


def print_correlation_by_columns(data, columns=None, method='pearson', target_column=None):
    """
    根据列名打印相关性信息
    
    参数:
    data: pandas DataFrame, 数据集
    columns: list, 需要分析的列名列表，默认为None表示使用所有数值列
    method: str, 计算相关性的方法 ('pearson', 'kendall', 'spearman')
    target_column: str, 可选的目标列名，如果指定则只显示与其他列的相关性
    
    返回:
    pandas.DataFrame: 相关性数据表
    """
    # 如果未指定列，则选择所有数值列
    if columns is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        # 确保目标列也在数据中
        if target_column and target_column not in columns:
            columns.append(target_column)
        numeric_data = data[columns]
    
    # 计算相关性矩阵
    corr_matrix = numeric_data.corr(method=method)
    
    # 如果指定了目标列，只显示与目标列的相关性
    if target_column:
        if target_column in corr_matrix.columns:
            target_corr = corr_matrix[target_column].drop(target_column)  # 删除自身相关性（为1）
            target_corr = target_corr.sort_values(key=abs, ascending=False)
            
            print(f"\n列 '{target_column}' 与其他特征的相关性 (方法: {method}):")
            print("=" * 50)
            for col, corr in target_corr.items():
                print(f"{col:30s}: {corr:8.3f}")
            
            # 转换为DataFrame返回
            result_df = pd.DataFrame({
                'feature': target_corr.index,
                'correlation': target_corr.values
            })
            return result_df
        else:
            print(f"警告: 列 '{target_column}' 在数据中未找到")
            return pd.DataFrame()
    else:
        # 显示完整相关性矩阵（控制台输出）
        print(f"\n特征相关性矩阵 (方法: {method}):")
        print("=" * 50)
        
        # 获取列名
        cols = corr_matrix.columns.tolist()
        
        # 打印表头
        header = f"{'':30s}" + "".join([f"{col:8.8s} " for col in cols])
        print(header)
        print("-" * len(header))
        
        # 打印每行数据
        for i, row_name in enumerate(cols):
            row_data = f"{row_name:30s}" + "".join([f"{corr_matrix.iloc[i, j]:8.3f} " for j in range(len(cols))])
            print(row_data)
        
        return corr_matrix


def plot_correlation_by_columns(data, columns=None, method='pearson', target_column=None, 
                               figsize=(10, 8), title=None):
    """
    根据列名绘制相关性图表
    
    参数:
    data: pandas DataFrame, 数据集
    columns: list, 需要分析的列名列表，默认为None表示使用所有数值列
    method: str, 计算相关性的方法 ('pearson', 'kendall', 'spearman')
    target_column: str, 可选的目标列名，如果指定则只显示该列与其他列的相关性
    figsize: tuple, 图形大小
    title: str, 图形标题，如果为None则自动生成
    
    返回:
    matplotlib.figure.Figure: 生成的图形对象
    """
    # 如果未指定列，则选择所有数值列
    if columns is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        # 确保目标列也在数据中
        if target_column and target_column not in columns:
            columns.append(target_column)
        numeric_data = data[columns]
    
    # 计算相关性矩阵
    corr_matrix = numeric_data.corr(method=method)
    
    # 如果指定了目标列，只显示与目标列的相关性
    if target_column and target_column in corr_matrix.columns:
        # 获取与目标列的相关性（排除自身）
        target_corr = corr_matrix[target_column].drop(target_column)
        target_corr = target_corr.sort_values(key=abs, ascending=False)
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 生成默认标题
        if title is None:
            title = f"'{target_column}' 与其他特征的相关性"
        
        # 绘制水平条形图
        colors = ['red' if x < 0 else 'blue' for x in target_corr.values]
        bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        
        # 设置标签和标题
        plt.yticks(range(len(target_corr)), target_corr.index)
        plt.xlabel(f'{method.capitalize()} 相关性系数')
        plt.title(title)
        plt.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, target_corr.values)):
            plt.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.2f}', 
                    va='center', ha='left' if value >= 0 else 'right')
        
        plt.tight_layout()
        
        return plt.gcf()
    else:
        # 绘制完整相关性矩阵热力图
        # 生成默认标题
        if title is None:
            title = f'特征相关性矩阵 ({method})'
        
        return plot_correlation_matrix(data, columns, method, figsize, True, 'coolwarm', title)


def plot_single_column_correlation(data, target_column, method='pearson', figsize=(10, 8)):
    """
    根据列名字符串绘制该列与其他所有列的相关性
    
    参数:
    data: pandas DataFrame, 数据集
    target_column: str, 目标列名字符串
    method: str, 计算相关性的方法 ('pearson', 'kendall', 'spearman')
    figsize: tuple, 图形大小
    
    返回:
    matplotlib.figure.Figure: 生成的图形对象
    """
    # 检查目标列是否存在
    if target_column not in data.columns:
        raise ValueError(f"列 '{target_column}' 在数据中未找到")
    
    # 选择所有数值列
    numeric_data = data.select_dtypes(include=[np.number])
    
    # 检查目标列是否为数值列
    if target_column not in numeric_data.columns:
        raise ValueError(f"列 '{target_column}' 不是数值列，无法计算相关性")
    
    # 计算与目标列的相关性
    correlations = {}
    target_data = data[target_column]
    
    for col in numeric_data.columns:
        if col != target_column:
            corr = data[col].corr(target_data, method=method)
            correlations[col] = corr
    
    # 转换为Series并排序
    if not correlations:
        raise ValueError(f"没有找到与 '{target_column}' 相关的其他数值列")
        
    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制水平条形图
    colors = ['red' if x < 0 else 'blue' for x in corr_series.values]
    bars = plt.barh(range(len(corr_series)), corr_series.values, color=colors, alpha=0.7)
    
    # 设置标签和标题
    plt.yticks(range(len(corr_series)), corr_series.index)
    plt.xlabel(f'{method.capitalize()} 相关性系数')
    plt.title(f"列 '{target_column}' 与其他数值特征的相关性")
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, corr_series.values)):
        plt.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.2f}', 
                va='center', ha='left' if value >= 0 else 'right')
    
    plt.tight_layout()
    
    return plt.gcf()


if __name__ == "__main__":
    # 加载数据
    data = Data.data
    
    # 示例1: 绘制所有数值特征的相关性矩阵
    #fig1 = plot_correlation_matrix(data)
    #plt.show()
    plot_single_column_correlation(data,"13号染色体的Z值")
    # 示例2: 绘制指定特征与目标变量的相关性
    # 注意：需要根据实际数据调整列名
    