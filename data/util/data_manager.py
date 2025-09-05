import pandas as pd
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
        
        directory = os.path.dirname(save_path)
       
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.data.to_csv(save_path, index=False, encoding='utf-8-sig')

    def get_col_count(self):
        """
        获取列数
        """
        return self.data.shape[1]

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
config = {
        "file_path": os.path.join(project_dir, "data", "cleaned_data.csv"),  
        
    }
Data = data_manager(config)