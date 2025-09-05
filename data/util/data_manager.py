import pandas as pd
import openpyxl
import os

class data_manager:
    def __init__(self, config):
        self.config = config
        self.data = data_manager.import_data(self.config["file_path"])
        
    def import_data(file_path):
        """
        数据导入
        """
        try:
            print(f"正在导入数据文件 {file_path}")
            # 根据文件扩展名选择正确的读取方法
            if file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                # 默认尝试读取，让pandas自动判断
                data = pd.read_csv(file_path)
        except Exception as e:
            print(f"导入数据文件 {file_path} 失败：{e}")
            return None
            
        print(f"数据文件 {file_path} 导入成功")
        return data
        
    def save_data(self, save_path):
        """
        数据保存
        """
        
        directory = os.path.dirname(save_path)
       
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.data.to_excel(save_path, index=False)

    def get_col_count(self):
        """
        获取列数
        """
        return self.data.shape[1]

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
# 检查文件是否存在，优先使用CSV格式
csv_file_path = os.path.join(project_dir, "data", "cleaned_data.csv")
xlsx_file_path = os.path.join(project_dir, "data", "cleaned_data.xlsx")

if os.path.exists(csv_file_path):
    file_path = csv_file_path
else:
    file_path = xlsx_file_path

config = {
        "file_path": file_path,  
        
    }
Data = data_manager(config)