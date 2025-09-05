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
            data = pd.read_excel(file_path)
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
config = {
        "file_path": os.path.join(project_dir, "data", "cleaned_data.xlsx"),  
        
    }
Data = data_manager(config)