def risk_factor(t):
    """
    风险因子的计算函数
    pram：t：检测孕日（单位：日）
    return：风险因子
    """
    t=t/7 
    if t<=12:
        y=1
    elif t<=27:
        y = 1+0.1*(t-12)
    else:
        R = 1+0.1*(27-12) 
        y = R*1.1**(t-27)
    return y

def diagnostic_efficacy_ratio(t,pt):
    """
    诊断效率比计算函数
    pram：t：检测孕日（单位：日）
    return：诊断效率比
    """
    return pt/risk_factor(t)
