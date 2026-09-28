import pandas as pd

col_names = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]

df = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None)
df.columns = col_names

print(df.shape)
print(df.head())

max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
max_cycles = max_cycles.rename(columns={'time_cycles': 'max_cycle'})

df = pd.merge(df, max_cycles, on='unit_number')
df['RUL'] = df['max_cycle'] - df['time_cycles']

print(df[['unit_number', 'time_cycles', 'max_cycle', 'RUL']].head(10))

sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
print(df[sensor_cols].std().sort_values())

print(df[['op_setting_1', 'op_setting_2', 'op_setting_3']].std())

constant_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
df = df.drop(columns=constant_sensors)

op_settings_to_drop = ['op_setting_1', 'op_setting_2', 'op_setting_3']
df = df.drop(columns=op_settings_to_drop)

print("Dropped sensors:", constant_sensors)
print("Dropped op settings:", op_settings_to_drop)
print("New shape:", df.shape)



import matplotlib.pyplot as plt

engine_1 = df[df['unit_number'] == 1]

fig, axes = plt.subplots(3, 1, figsize=(8, 8))
axes[0].plot(engine_1['time_cycles'], engine_1['sensor_9'])
axes[0].set_title('sensor_9 over time, engine 1')

axes[1].plot(engine_1['time_cycles'], engine_1['sensor_14'])
axes[1].set_title('sensor_14 over time, engine 1')

axes[2].plot(engine_1['time_cycles'], engine_1['sensor_4'])
axes[2].set_title('sensor_4 over time, engine 1')

plt.tight_layout()
plt.savefig('degradation_check.png')
print("Saved degradation_check.png")


window_size = 5
surviving_sensors = [col for col in df.columns if col.startswith('sensor_')]

df = df.sort_values(['unit_number', 'time_cycles'])

for col in surviving_sensors:
    df[f'{col}_rollmean'] = df.groupby('unit_number')[col].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean())
    df[f'{col}_rollstd'] = df.groupby('unit_number')[col].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std())

df[[f'{c}_rollstd' for c in surviving_sensors]] = df[[f'{c}_rollstd' for c in surviving_sensors]].fillna(0)

print(df.shape)
print(df[['sensor_4', 'sensor_4_rollmean', 'sensor_4_rollstd']].head(10))

df.to_csv('data/processed_FD001.csv', index=False)
print("Saved processed_FD001.csv")