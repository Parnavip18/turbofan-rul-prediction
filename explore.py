import pandas as pd 

df = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None) 
print(df.shape) 
print(df.head()) 

col_names = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)] 
df.columns = col_names 
print(df.head())

lifespans = df.groupby('unit_number')['time_cycles'].max() 
print(lifespans.head(10)) 
print(lifespans.describe())