import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    KM_AVAILABLE = True
except ImportError:
    KM_AVAILABLE = False
    print("警告: 未安装 lifelines 库，KM模型不可用。请使用 'pip install lifelines' 安装。")

class KMModel:
    def __init__(self):
        '''初始化Kaplan-Meier模型'''
        if not KM_AVAILABLE:
            raise ImportError("lifelines 库未安装，无法使用KM模型")
            
        self.model = KaplanMeierFitter()
        self.is_fitted = False
        self.groups = {}  # 存储不同组的模型

    def fit(self, durations, event_observed, label=None):
        '''拟合Kaplan-Meier模型
        Args:
            durations: 生存时间
            event_observed: 事件观察指示器（1表示事件发生，0表示删失）
            label: 模型标签
        '''
        self.model.fit(durations, event_observed, label=label)
        self.is_fitted = True

    def fit_groups(self, durations, event_observed, groups, group_names=None):
        '''按组别拟合多个Kaplan-Meier模型
        Args:
            durations: 生存时间
            event_observed: 事件观察指示器
            groups: 分组标识符
            group_names: 组名（可选）
        '''
        unique_groups = np.unique(groups)
        for i, group in enumerate(unique_groups):
            group_mask = (groups == group)
            group_durations = durations[group_mask]
            group_events = event_observed[group_mask]
            group_name = group_names[i] if group_names and i < len(group_names) else str(group)
            
            kmf = KaplanMeierFitter()
            kmf.fit(group_durations, group_events, label=group_name)
            self.groups[group] = kmf
        
        self.is_fitted = True

    def predict(self, times):
        '''预测生存概率
        Args:
            times: 时间点
        Returns:
            生存概率
        '''
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        return self.model.predict(times)

    def predict_groups(self, times, group_labels=None):
        '''预测各组的生存概率
        Args:
            times: 时间点
            group_labels: 要预测的组标签列表（可选）
        Returns:
            各组生存概率字典
        '''
        if not self.is_fitted or not self.groups:
            raise ValueError("组模型尚未拟合，请先调用fit_groups方法")
            
        result = {}
        labels = group_labels if group_labels else self.groups.keys()
        for label in labels:
            if label in self.groups:
                result[label] = self.groups[label].predict(times)
        return result

    def plot(self, ax=None, ci_show=True, ci_alpha=0.3, **kwargs):
        '''绘制Kaplan-Meier曲线
        Args:
            ax: matplotlib轴对象（可选）
            ci_show: 是否显示置信区间
            ci_alpha: 置信区间透明度
        '''
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
        self.model.plot_survival_function(ax=ax, ci_show=ci_show, ci_alpha=ci_alpha, **kwargs)
        ax.set_xlabel('时间')
        ax.set_ylabel('生存概率')
        ax.set_title('Kaplan-Meier生存曲线')
        ax.grid(True, alpha=0.3)
        return ax

    def plot_groups(self, ax=None, ci_show=False, **kwargs):
        '''绘制各组的Kaplan-Meier曲线
        Args:
            ax: matplotlib轴对象（可选）
            ci_show: 是否显示置信区间
        '''
        if not self.is_fitted or not self.groups:
            raise ValueError("组模型尚未拟合，请先调用fit_groups方法")
            
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
        for group, model in self.groups.items():
            model.plot_survival_function(ax=ax, ci_show=ci_show, **kwargs)
            
        ax.set_xlabel('时间')
        ax.set_ylabel('生存概率')
        ax.set_title('Kaplan-Meier生存曲线（按组别）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def add_at_risk_counts(self, ax, groups=None):
        '''在图中添加风险人数
        Args:
            ax: matplotlib轴对象
            groups: 要显示的组（可选）
        '''
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        if groups:
            add_at_risk_counts(*groups, ax=ax)
        elif self.groups:
            add_at_risk_counts(*self.groups.values(), ax=ax)
        else:
            add_at_risk_counts(self.model, ax=ax)

    def get_survival_function(self):
        '''获取生存函数
        Returns:
            生存函数
        '''
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        return self.model.survival_function_

    def get_confidence_interval(self):
        '''获取置信区间
        Returns:
            置信区间
        '''
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        return self.model.confidence_interval_

    def get_median_survival_time(self):
        '''获取中位生存时间
        Returns:
            中位生存时间
        '''
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        return self.model.median_survival_time_

    def get_confidence_interval_of_median_survival_time(self):
        '''获取中位生存时间的置信区间
        Returns:
            中位生存时间的置信区间
        '''
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        return self.model.confidence_interval_survival_function_

    def analyze_km_results(self, km_fitters, format_gestational_age_func):
        '''分析KM结果，计算中位生存时间和10%未达标时间
        Args:
            km_fitters: KM拟合器字典
            format_gestational_age_func: 格式化孕周的函数
        Returns:
            分析结果
        '''
        results = {}
        for cluster_id in km_fitters.keys():
            kmf = km_fitters[cluster_id]
            median_time = kmf.median_survival_time_
            if not np.isnan(median_time):
                print(f"簇 {cluster_id} 中位生存时间: {format_gestational_age_func(median_time)} (原始值: {median_time:.2f}天)")
            else:
                print(f"簇 {cluster_id} 中位生存时间: 无法计算（数据不足）")
            
            # 计算90%分位数时间（即10%的孕妇仍未达标的时间）
            # 这里我们计算生存概率降到0.1的时间点
            try:
                # 获取生存概率和时间
                survival_prob = kmf.survival_function_.values.flatten()
                times = kmf.survival_function_.index
                
                # 找到生存概率降到0.1的时间点
                # 注意：KM曲线是递减的，所以我们找第一个小于等于0.1的点
                below_10pct = survival_prob <= 0.1
                if np.any(below_10pct):
                    idx = np.where(below_10pct)[0][0]
                    time_10pct = times[idx]
                    print(f"簇 {cluster_id} 10%未达标时间: {format_gestational_age_func(time_10pct)} (原始值: {time_10pct:.2f}天)")
                    results[cluster_id] = {
                        'median_time': median_time,
                        'time_10pct': time_10pct
                    }
                else:
                    print(f"簇 {cluster_id} 10%未达标时间: 超出观察范围")
                    results[cluster_id] = {
                        'median_time': median_time,
                        'time_10pct': None
                    }
            except Exception as e:
                print(f"簇 {cluster_id} 10%未达标时间: 计算失败 ({e})")
                results[cluster_id] = {
                    'median_time': median_time,
                    'time_10pct': None
                }
        
        return results

if __name__ == "__main__":
    if KM_AVAILABLE:
        print("KM模型已定义，可以用于生存分析任务")
        print("使用 lifelines 库的 KaplanMeierFitter 实现")
    else:
        print("请安装lifelines库以使用KM模型: pip install lifelines")