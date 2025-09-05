import pandas as pd
import openpyxl
import os

class data_cleaner:
    def __init__(self, config):
        self.config = config
        self.data = data_cleaner.import_data(self.config["file_path"])
        

    
    def scan_and_fix_column(self,col,fill_value):
    def scan_and_fix_column(self,col,fill_value):
        """
        param:
        col: 列号
        """
        
       
        self.data.iloc[:, col].fillna(fill_value, inplace=True)
        self.data.iloc[:, col].fillna(fill_value, inplace=True)

    
    def limit_precision(self,col:int, precision:int):
        """
        数据数度限制
        param:
        col: 列号
        precision: 保留的小数位数
        
        """
        
        self.data.iloc[:, col] = self.data.iloc[:, col].round(precision)

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

    def get_col_count(self):
        """
        获取列数
        """
        return self.data.shape[1]

if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    config = {
        "file_path": os.path.join(project_dir, "data", "data.xlsx"),  
        "save_path": os.path.join(project_dir, "data", "cleaned_data.xlsx"),  
    }

    cleaner = data_cleaner(config)
    # cleaner.save_data(config["save_path"])