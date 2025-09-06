import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re

class data_cleaner:
    def __init__(self, config):
        self.config = config
        self.data = data_cleaner.import_data(self.config["file_path"])
        

    
    def scan_and_fix_column(self,col,fill_value):
        """
        param:
        col: 列号
        """
        
       
        self.data.iloc[:, col].fillna(fill_value, inplace=True)

    
    def limit_precision(self,col:int, precision:int):
        """
        数据数度限制
        param:
        col: 列号
        precision: 保留的小数位数
        
        """
        
        self.data.iloc[:, col] = self.data.iloc[:, col].round(precision)

    def check_missing_values(self):
        """
        检查数据中的缺失值
        返回每列的缺失值数量和比例
        """
        if self.data is None:
            print("数据未加载")
            return None
        
        # 使用df.isnull()检查缺失值
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        
        # 创建结果DataFrame
        missing_info = pd.DataFrame({
            'Missing_Count': missing_values,
            'Missing_Percentage': missing_percentage
        })
        
        return missing_info
    
    def unified_clean_process(self):
        """
        统一的数据清理方法，遍历每一行，检查每一列的值并进行相应处理
        """
        if self.data is None:
            print("数据未加载")
            return
        
        print("开始执行统一数据清理过程...")
        
        # 获取所有列名
        columns = self.data.columns.tolist()
        
        # 遍历每一行
        for index, row in self.data.iterrows():
            # 遍历每一列
            for col_name in columns:
                # 检查当前列的值是否有问题并进行相应处理
                self._process_column_value(index, col_name, row[col_name])
        
        print("统一数据清理过程完成")
    
    def _process_column_value(self, index, col_name, value):
        """
        处理特定列的值
        param:
        index: 行索引
        col_name: 列名
        value: 当前值
        """
        # 根据列名和值的特点进行不同的处理
        if col_name == '末次月经' and pd.isna(value):
            # 处理末次月经缺失值
            self._calculate_lmp_for_row(index)
        elif col_name == '末次月经' and not pd.isna(value):
            # 检查并统一末次月经数据格式
            self._standardize_lmp_format(index, value)

        elif col_name == '检测日期' and not pd.isna(value):
            # 检查并统一检测日期数据格式
            self._standardize_exam_date_format(index, value)
        elif col_name == '检测孕周' and not pd.isna(value):
            # 检查并统一检测孕周数据格式
            self._standardize_gestational_week_format(index, value)
        elif col_name == '怀孕次数' and not pd.isna(value):
            # 处理字符串格式的"≥3"
            if isinstance(value, str) and value == '≥3':
                self.data.at[index, col_name] = 3
        elif col_name == '胎儿是否健康' and not pd.isna(value):
            # 处理胎儿是否健康列，将'是'转换为1，'否'转换为0
            if value == '是':
                self.data.at[index, col_name] = 1
            elif value == '否':
                self.data.at[index, col_name] = 0
        # 可以继续添加其他列的处理逻辑
    
    def _standardize_gestational_week_format(self, index, value):
        """
        统一检测孕周数据格式为以天为单位
        param:
        index: 行索引
        value: 当前值
        """
        # 处理类似"15w+6"或"13w"的格式，w可能是大写或小写
        if isinstance(value, str):
            # 使用正则表达式匹配孕周格式，支持大小写W
            weeks_match = re.match(r'(\d+)([wW])(?:\+(\d+))?', value)
            if weeks_match:
                weeks = int(weeks_match.group(1))
                days = int(weeks_match.group(3)) if weeks_match.group(3) else 0
                total_days = weeks * 7 + days
                self.data.at[index, '检测孕周'] = total_days
        # 可以添加其他孕周格式的处理逻辑
    
    def _standardize_exam_date_format(self, index, value):
        """
        统一检测日期数据格式为 YYYY/MM/DD
        param:
        index: 行索引
        value: 当前值
        """
        # 检查是否为20230429格式
        if isinstance(value, (int, str)) and len(str(value)) == 8 and str(value).isdigit():
            try:
                # 解析20230429格式的日期
                parsed_date = datetime.strptime(str(value), '%Y%m%d')
                # 转换为YYYY/MM/DD格式
                self.data.at[index, '检测日期'] = parsed_date.strftime('%Y/%m/%d')
            except ValueError:
                # 如果解析失败，保持原值
                pass
        # 可以添加其他日期格式的处理逻辑
    
    def _standardize_lmp_format(self, index, value):
        """
        统一末次月经数据格式为 YYYY/MM/DD
        param:
        index: 行索引
        value: 当前值
        """
        # 尝试解析不同格式的日期
        date_formats = ['%Y/%m/%d', '%Y-%m-%d', '%Y/%#m/%#d', '%Y-%#m-%#d']
        parsed_date = None
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(str(value), fmt)
                break
            except ValueError:
                continue
        
        # 如果成功解析，转换为统一格式
        if parsed_date:
            self.data.at[index, '末次月经'] = parsed_date.strftime('%Y/%m/%d')
    
    def _calculate_lmp_for_row(self, index):
        """
        为特定行计算末次月经日期
        param:
        index: 行索引
        """
        row = self.data.loc[index]
        exam_date = row['检测日期']
        gestational_week = row['检测孕周']
        
        # 解析检测日期
        if isinstance(exam_date, str):
            exam_date = datetime.strptime(exam_date, '%Y%m%d')
        elif isinstance(exam_date, int):
            exam_date = datetime.strptime(str(exam_date), '%Y%m%d')
        
        # 解析孕周
        # 现在孕周已经被转换为以天为单位的数值，直接使用
        if isinstance(gestational_week, (int, float)):
            total_days = gestational_week
        else:
            # 如果不是数值类型，尝试解析字符串格式
            if isinstance(gestational_week, str):
                # 处理类似"13w+6"或"13w"的格式
                weeks_match = re.match(r'(\d+)w(?:\+(\d+))?', gestational_week)
                if weeks_match:
                    weeks = int(weeks_match.group(1))
                    days = int(weeks_match.group(2)) if weeks_match.group(2) else 0
                    total_days = weeks * 7 + days
                else:
                    # 如果无法解析，跳过这一行
                    return
            else:
                # 如果无法识别的类型，跳过这一行
                return
        
        # 计算末次月经日期
        lmp_date = exam_date - timedelta(days=total_days)
        
        # 更新数据
        self.data.at[index, '末次月经'] = lmp_date.strftime('%Y/%m/%d')
    
    def plot_missing_values_heatmap(self, save_path=None):
        """
        使用seaborn heatmap可视化缺失值分布
        param:
        save_path: 可选，保存图片的路径
        """
        if self.data is None:
            print("数据未加载，无法生成热力图")
            return
        
        # 创建缺失值掩码
        plt.figure(figsize=(12, 8))
        missing_mask = self.data.isnull()
        
        # 生成热力图
        sns.heatmap(missing_mask, cbar=True, yticklabels=False, cmap='viridis')
        plt.title('数据缺失值分布热力图')
        plt.xlabel('列 (字段)')
        plt.ylabel('行 (样本)')
        
        # 保存或显示图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"缺失值热力图已保存至 {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def import_data(file_path):
        """
        数据导入
        """
        try:
            print(f"正在导入数据文件 {file_path}")
            # 根据文件扩展名自动选择读取方式
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
        except Exception as e:
            print(f"导入数据文件 {file_path} 失败：{e}")
            return None
        print(f"数据文件 {file_path} 导入成功")
        return data
    
    def save_data(self, save_path):
        """
        数据保存
        """
        # 检查数据是否已加载
        if self.data is None:
            print("无法保存数据：数据未加载")
            return False
        
        directory = os.path.dirname(save_path)
       
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        # 保存为CSV格式
        self.data.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存至 {save_path}")
        return True

    def get_col_count(self):
        """
        获取列数
        """
        if self.data is None:
            print("数据未加载")
            return None
        return self.data.shape[1]
    
    def date_standardization(self):
        """
        数据日期统一处理,按照YYYY/MM/DD格式
        """
        if self.data is None:
            print("数据未加载")
            return
        
        # 创建日期转换字典
        date_transforms = {
            '检测日期': self._standardize_exam_date_format,
            '末次月经': self._standardize_lmp_format,
            '检测孕周': self._standardize_gestational_week_format
        }
        
        # 遍历数据，对每一行应用日期转换
        for index, row in self.data.iterrows():
            for col_name, transform_func in date_transforms.items():
                if col_name in self.data.columns and pd.notna(row[col_name]):
                    # 调用对应的转换函数处理日期
                    transform_func(index, row[col_name])
if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    config = {
        "file_path": os.path.join(project_dir, "data", "男胎检测数据.xlsx"),  
        "save_path": os.path.join(project_dir, "data", "date_cleaned_data_male.xlsx"),  
    }

    # 创建数据清洗器实例
    cleaner = data_cleaner(config)
    
    # 检查初始缺失值
    missing_info = cleaner.check_missing_values()
    if missing_info is not None:
        print("\n初始缺失值检查结果：")
        print(missing_info)
    
    # 生成处理前的缺失值热力图，用于与清理后的热力图对比
    heatmap_before_save_path = os.path.join(project_dir, "data", "male_missing_values_heatmap_before.png")
    cleaner.plot_missing_values_heatmap(save_path=heatmap_before_save_path)
    print(f"处理前的缺失值热力图已保存至 {heatmap_before_save_path}")
    
    # 执行统一的数据清理过程
    cleaner.unified_clean_process()
    cleaner.date_standardization()
    # 再次检查缺失值，确认清理效果
    missing_info = cleaner.check_missing_values()
    if missing_info is not None:
        print("\n数据清理后的缺失值检查结果：")
        print(missing_info)
    
    # 生成缺失值热力图，可视化数据质量
    heatmap_save_path = os.path.join(project_dir, "data", "male_missing_values_heatmap.png")
    cleaner.plot_missing_values_heatmap(save_path=heatmap_save_path)
    
    # 筛选GC含量在40%到60%之间的数据
    if 'GC含量' in cleaner.data.columns:
        original_count = len(cleaner.data)
        cleaner.data = cleaner.data[(cleaner.data['GC含量'] >= 0.4) & (cleaner.data['GC含量'] <= 0.6)]
        filtered_count = len(cleaner.data)
        print(f"根据GC含量(40%-60%)筛选数据，原始数据量: {original_count}，筛选后数据量: {filtered_count}")
    else:
        print("警告: 数据中未找到'GC含量'列")
    
    # 保存清洗后的数据
    if cleaner.data is not None:
        cleaner.save_data(config["save_path"])