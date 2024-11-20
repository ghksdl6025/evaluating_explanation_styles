import pandas as pd
import scipy
import scipy.stats
import numpy as np
import math
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

df = pd.read_excel('./real_analysi_data.xlsx',sheet_name='Sheet1')
# print(df)

higher_cat = [i.split(' ')[0] for i in list(df['Category'])]
df['Higher Category'] = higher_cat
# print(list(df['Higher Category']))

# Function to calculate eta-squared
def calculate_eta_squared(anova_table):
    sum_sq_method = anova_table.loc['C(Method)', 'sum_sq']
    sum_sq_total = sum(anova_table['sum_sq'])
    eta_squared = sum_sq_method / sum_sq_total
    return eta_squared

def calculate_partial_eta_squared(anova_table):
    partial_eta_squared = {}
    for effect in ['C(IV1)', 'C(IV2)', 'C(IV1):C(IV2)']:
        sum_sq_effect = anova_table.loc[effect, 'sum_sq']
        sum_sq_effect_and_residual = sum_sq_effect + anova_table.loc['Residual', 'sum_sq']
        partial_eta_squared[effect] = sum_sq_effect / sum_sq_effect_and_residual
    return partial_eta_squared


def h1():
    ########
    ## H1 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H1')
    task_perf = list(df['Before XAI Sum of task performance'])
    task_perf_h = []
    task_perf_l = []
    for pos, i in enumerate(higher_cat):
        if i == 'High':
            task_perf_h.append(task_perf[pos])
        elif i == 'Low':
            task_perf_l.append(task_perf[pos])

    ttest_result = scipy.stats.ttest_ind(task_perf_h, task_perf_l)
    mean_diff = np.average(task_perf_h) - np.average(task_perf_l)
    print(mean_diff)
    print(ttest_result)

    # Cohen's d
    pooled_std = math.sqrt(((len(task_perf_h) - 1) * np.std(task_perf_h, ddof=1) ** 2 +
                            (len(task_perf_l) - 1) * np.std(task_perf_l, ddof=1) ** 2) / (len(task_perf_h) + len(task_perf_l) - 2))
    cohen_d = mean_diff / pooled_std
    print(cohen_d)

    print('\n')


def h2():
    ########
    ## H2 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence

    print('Testing H2')
    print('Whole accuracy')
    exp = [' '.join(i.split(' ')[1:]) for i in list(df['Category'])]
    df['Exp'] = exp

    task_perf = list(df['After XAI Sum of task performance'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)
    
    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)

    mean_f1 = np.mean(task_perf_fi)
    mean_r = np.mean(task_perf_r)
    mean_cf = np.mean(task_perf_cf)
    print(f"Mean of Group task performance: {mean_f1:.4f}")
    print(f"Mean of Group rule: {mean_r:.4f}")
    print(f"Mean of Group counterfactual: {mean_cf:.4f}")
    print('\n')

    print('High accuracy')
    dft = df[df['Higher Category'] =='High']
    exp = [' '.join(i.split(' ')[1:]) for i in list(dft['Category'])]
    dft['Exp'] = exp

    task_perf = list(dft['After XAI Sum of task performance'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)


    print('Low accuracy')
    dft = df[df['Higher Category'] =='Low']
    exp = [' '.join(i.split(' ')[1:]) for i in list(dft['Category'])]
    dft['Exp'] = exp

    task_perf = list(dft['After XAI Sum of task performance'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)

    print('\n')

def h3():
    ########
    ## H3 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H3')
    exp = [' '.join(i.split(' ')[1:]) for i in list(df['Category'])]
    df['Exp'] = exp

    print('Full paired T-test')
    before_task_perf = np.array(list(df['Before XAI Sum of task performance']))
    after_task_perf = np.array(list(df['After XAI Sum of task performance']))

    ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
    mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
    print(ttest_result)
    # Cohen's d
    diff = after_task_perf - before_task_perf
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    print(f"Mean difference: {mean_diff}")        
    print(f"Std difference: {std_diff}")        
    cohen_d = mean_diff / std_diff
    print(cohen_d)

    print('\n')

    print('Group by Accuracy')
    groups = df.groupby(['Higher Category'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of task performance']))
        after_task_perf = np.array(list(group['After XAI Sum of task performance']))
        
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')


    print('Group by Accuracy and Explanation style')
    groups = df.groupby(['Higher Category', 'Exp'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of task performance']))
        after_task_perf = np.array(list(group['After XAI Sum of task performance']))
        
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')

    print('Group by Explanation style')
    groups = df.groupby(['Exp'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of task performance']))
        after_task_perf = np.array(list(group['After XAI Sum of task performance']))
        
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')

def h4():
    ########
    ## H4 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H4')
    task_perf = list(df['Before XAI Sum of agreement'])
    task_perf_h = []
    task_perf_l = []
    for pos, i in enumerate(higher_cat):
        if i == 'High':
            task_perf_h.append(task_perf[pos])
        elif i == 'Low':
            task_perf_l.append(task_perf[pos])

    ttest_result = scipy.stats.ttest_ind(task_perf_h, task_perf_l)
    mean_diff = np.average(task_perf_h) - np.average(task_perf_l)
    print(mean_diff)
    print(ttest_result)
    # Cohen's d
    pooled_std = math.sqrt(((len(task_perf_h) - 1) * np.std(task_perf_h, ddof=1) ** 2 +
                            (len(task_perf_l) - 1) * np.std(task_perf_l, ddof=1) ** 2) / (len(task_perf_h) + len(task_perf_l) - 2))
    cohen_d = mean_diff / pooled_std
    print(cohen_d)

    print('\n')

def h5():
    ########
    ## H5 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H5')
    print('Whole accuracy')
    exp = [' '.join(i.split(' ')[1:]) for i in list(df['Category'])]
    df['Exp'] = exp

    task_perf = list(df['After XAI Sum of agreement'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")
    print(round(np.mean(task_perf_fi), 2),
          round(np.mean(task_perf_r), 2),
          round(np.mean(task_perf_cf), 2))

    all_data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    overall_mean = np.mean(all_data)

    # Calculate SS_total
    ss_total = np.sum((all_data - overall_mean) ** 2)

    # Calculate SS_between
    ss_between = sum(len(group) * (np.mean(group) - overall_mean) ** 2 for group in [task_perf_fi, task_perf_r, task_perf_cf])

    # Calculate SS_within (Residuals)
    ss_within = sum(np.sum((group - np.mean(group)) ** 2) for group in [task_perf_fi, task_perf_r, task_perf_cf])

    # Degrees of freedom
    df_between = 2  # Number of groups - 1
    df_within = len(all_data) - 3  # Total observations - Number of groups
    ms_within = ss_within / df_within
    eta_squared = ss_between / ss_total
    print(f"Eta-Squared: {eta_squared:.3f}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)

    mean_f1 = np.mean(task_perf_fi)
    mean_r = np.mean(task_perf_r)
    mean_cf = np.mean(task_perf_cf)
    print(f"Mean of Group task performance: {mean_f1:.4f}")
    print(f"Mean of Group rule: {mean_r:.4f}")
    print(f"Mean of Group counterfactual: {mean_cf:.4f}")

    print('\n')

    print('High accuracy')
    dft = df[df['Higher Category'] =='High']
    exp = [' '.join(i.split(' ')[1:]) for i in list(dft['Category'])]
    dft['Exp'] = exp

    task_perf = list(dft['After XAI Sum of agreement'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)


    print('Low accuracy')
    dft = df[df['Higher Category'] =='Low']
    exp = [' '.join(i.split(' ')[1:]) for i in list(dft['Category'])]
    dft['Exp'] = exp

    task_perf = list(dft['After XAI Sum of agreement'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)

    print('\n')

def h6():
    ########
    ## H6 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H6')
    exp = [' '.join(i.split(' ')[1:]) for i in list(df['Category'])]
    df['Exp'] = exp

    print('Full paired T-test')
    before_task_perf = np.array(list(df['Before XAI Sum of agreement']))
    after_task_perf = np.array(list(df['After XAI Sum of agreement']))

    ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
    mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
    print(ttest_result)
    # Cohen's d
    diff = after_task_perf - before_task_perf
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    print(f"Mean difference: {mean_diff}")        
    print(f"Std difference: {std_diff}")        
    cohen_d = mean_diff / std_diff
    print(cohen_d)

    print('\n')

    print('Group by Accuracy')
    groups = df.groupby(['Higher Category'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of agreement']))
        after_task_perf = np.array(list(group['After XAI Sum of agreement']))
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')

    print('Group by Accuracy and Explanation style')
    groups = df.groupby(['Higher Category', 'Exp'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of agreement']))
        after_task_perf = np.array(list(group['After XAI Sum of agreement']))
        
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')

    print('Group by Explanation style')
    groups = df.groupby(['Exp'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of agreement']))
        after_task_perf = np.array(list(group['After XAI Sum of agreement']))
        
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')

def h7():
    ########
    ## H7 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H7')
    task_perf = list(df['Before XAI Sum of decision confidence'])
    task_perf_h = []
    task_perf_l = []
    for pos, i in enumerate(higher_cat):
        if i == 'High':
            task_perf_h.append(task_perf[pos])
        elif i == 'Low':
            task_perf_l.append(task_perf[pos])

    ttest_result = scipy.stats.ttest_ind(task_perf_h, task_perf_l)
    mean_diff = np.average(task_perf_h) - np.average(task_perf_l)
    print(mean_diff)
    print(ttest_result)
    # print(task_perf_l)
    # Cohen's d
    pooled_std = math.sqrt(((len(task_perf_h) - 1) * np.std(task_perf_h, ddof=1) ** 2 +
                            (len(task_perf_l) - 1) * np.std(task_perf_l, ddof=1) ** 2) / (len(task_perf_h) + len(task_perf_l) - 2))
    cohen_d = mean_diff / pooled_std
    print(cohen_d)

def h8():
    ########
    ## H8 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H8')
    print('Whole accuracy')
    exp = [' '.join(i.split(' ')[1:]) for i in list(df['Category'])]
    df['Exp'] = exp

    task_perf = list(df['After XAI Sum of decision confidence'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)

    mean_f1 = np.mean(task_perf_fi)
    mean_r = np.mean(task_perf_r)
    mean_cf = np.mean(task_perf_cf)
    print(f"Mean of Group task performance: {mean_f1:.4f}")
    print(f"Mean of Group rule: {mean_r:.4f}")
    print(f"Mean of Group counterfactual: {mean_cf:.4f}")
    print('\n')

    print('High accuracy')
    dft = df[df['Higher Category'] =='High']
    exp = [' '.join(i.split(' ')[1:]) for i in list(dft['Category'])]
    dft['Exp'] = exp

    task_perf = list(dft['After XAI Sum of decision confidence'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)


    print('Low accuracy')
    dft = df[df['Higher Category'] =='Low']
    exp = [' '.join(i.split(' ')[1:]) for i in list(dft['Category'])]
    dft['Exp'] = exp

    task_perf = list(dft['After XAI Sum of decision confidence'])
    task_perf_fi = []
    task_perf_r = []
    task_perf_cf = []
    for pos, i in enumerate(exp):
        
        if i == 'Feature Importance':
            task_perf_fi.append(task_perf[pos])
        elif i == 'Rule':
            task_perf_r.append(task_perf[pos])
        elif i == 'Counterfactual':
            task_perf_cf.append(task_perf[pos])

    f_stat, p_value = scipy.stats.f_oneway(task_perf_fi, task_perf_r, task_perf_cf)

    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_value}")

    # Degree of freedom
    k = 3  # 그룹의 수
    N = len(task_perf_fi) + len(task_perf_r) + len(task_perf_cf)  # 전체 샘플 수
    df_between = k - 1  # 모델 자유도
    df_within = N - k  # 오차 자유도

    print(f"Degrees of freedom (between groups): {df_between}")
    print(f"Degrees of freedom (within groups): {df_within}")

    # Tukey HSD 사후검정 수행
    print('Tukey HSD Post hoc analysis')
    data = np.concatenate([task_perf_fi, task_perf_r, task_perf_cf])
    groups = ['Group 1'] * len(task_perf_fi) + ['Group 2'] * len(task_perf_r) + ['Group 3'] * len(task_perf_cf)

    tukey_result = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
    print(tukey_result)

    print('\n')    

def h9():
    ########
    ## H9 ##
    ########
    # After XAI Sum of task performance
    # Before XAI Sum of agreement
    # Before XAI Sum of decision confidence
    print('Testing H9')
    exp = [' '.join(i.split(' ')[1:]) for i in list(df['Category'])]
    df['Exp'] = exp

    print('Full paired T-test')
    before_task_perf = np.array(list(df['Before XAI Sum of decision confidence']))
    after_task_perf = np.array(list(df['After XAI Sum of decision confidence']))

    ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
    mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
    print(ttest_result)
    # Cohen's d
    diff = after_task_perf - before_task_perf
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    print(f"Mean difference: {mean_diff}")        
    print(f"Std difference: {std_diff}")        
    cohen_d = mean_diff / std_diff
    print(cohen_d)

    print('\n')

    print('Group by Accuracy')
    groups = df.groupby(['Higher Category'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of decision confidence']))
        after_task_perf = np.array(list(group['After XAI Sum of decision confidence']))
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')
    print('Group by Accuracy and Explanation style')
    groups = df.groupby(['Higher Category', 'Exp'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of decision confidence']))
        after_task_perf = np.array(list(group['After XAI Sum of decision confidence']))
        
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')

    print('Group by Explanation style')
    groups = df.groupby(['Exp'])
    for _, group in groups:
        print(_)
        before_task_perf = np.array(list(group['Before XAI Sum of decision confidence']))
        after_task_perf = np.array(list(group['After XAI Sum of decision confidence']))
        
        ttest_result = scipy.stats.ttest_rel(before_task_perf, after_task_perf)
        mean_diff = np.average(before_task_perf) - np.average(after_task_perf)
        print(ttest_result)

        # Cohen's d
        diff = after_task_perf - before_task_perf
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        print(f"Mean difference: {mean_diff}")        
        print(f"Std difference: {std_diff}")        
        cohen_d = mean_diff / std_diff
        print(cohen_d)

        print('\n')

if __name__ == '__main__':
    h6()