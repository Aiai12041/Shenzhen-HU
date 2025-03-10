import numpy as np
import pandas as pd
from pysal.lib import weights
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam

# 设置随机种子确保结果可复现
np.random.seed(42)

# 定义区域数量
regions = 100

# 模拟数据生成
time_range = pd.date_range('2017-06', '2019-06', freq='M')

# 生成动态PPI数据（面板数据）
ppi_data = {
    'region_id': np.repeat(np.arange(regions), len(time_range)),
    'date': np.tile(time_range, regions),
    'PPI': np.random.normal(100, 10, regions*len(time_range))
}

# 生成静态HSI/HHI数据（横截面数据）
static_data = {
    'region_id': np.arange(regions),
    'HSI': np.random.uniform(0, 1, regions),
    'HHI': np.random.uniform(0.2, 0.8, regions)
}

# 创建数据框
df_dynamic = pd.DataFrame(ppi_data)
df_static = pd.DataFrame(static_data)

# 创建空间权重矩阵（示例）
coordinates = np.random.rand(regions, 2)
w = weights.KNN(coordinates, k=4)
w.transform = 'r'

# 计算静态变量的空间滞后
static_vars = ['HSI', 'HHI']
for var in static_vars:
    df_static[f'W_{var}'] = weights.spatial_lag.lag_spatial(w, df_static[var].values.reshape(-1,1))

# 合并动态静态数据
full_data = df_dynamic.merge(df_static, on='region_id')

# 添加时间编码
full_data['month'] = full_data['date'].dt.month
full_data['month_sin'] = np.sin(2 * np.pi * full_data['month']/12)
full_data['month_cos'] = np.cos(2 * np.pi * full_data['month']/12)

# 定义特征
dynamic_features = ['PPI', 'month_sin', 'month_cos']  # 随时间变化的特征
static_features = ['HSI', 'HHI', 'W_HSI', 'W_HHI']  # 静态特征

# 数据标准化
scaler_dynamic = StandardScaler()
scaler_static = StandardScaler()

# 分别标准化动态和静态特征
full_data[dynamic_features] = scaler_dynamic.fit_transform(full_data[dynamic_features])
full_data[static_features] = scaler_static.fit_transform(full_data[static_features])

# 构建时空序列数据
def create_sequences(data, region_ids, time_steps):
    X_dynamic, X_static, y = [], [], []
    for r in region_ids:
        region_data = data[data['region_id'] == r].sort_values('date')
        static_values = region_data[static_features].iloc[0]  # 静态特征只取一次
        for i in range(len(region_data)-time_steps):
            X_dynamic.append(region_data[dynamic_features].values[i:i+time_steps])
            X_static.append(static_values.values)
            y.append(region_data['PPI'].values[i+time_steps])
    return np.array(X_dynamic), np.array(X_static), np.array(y)

# 创建训练序列
time_steps = 6
X_dyn, X_stat, y = create_sequences(full_data, np.arange(regions), time_steps)

print(f"数据形状: X_dynamic={X_dyn.shape}, X_static={X_stat.shape}, y={y.shape}")

# 构建混合模型
# 动态特征分支
dynamic_input = Input(shape=(time_steps, len(dynamic_features)))
lstm = LSTM(64, return_sequences=True)(dynamic_input)
lstm_out = LSTM(32)(lstm)

# 静态特征分支
static_input = Input(shape=(len(static_features),))
static_dense = Dense(16, activation='relu')(static_input)

# 特征融合
merged = concatenate([lstm_out, static_dense])
dense1 = Dense(32, activation='relu')(merged)
output = Dense(1)(dense1)

# 创建模型
model = Model(inputs=[dynamic_input, static_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print(model.summary())

# 模型训练
history = model.fit(
    [X_dyn, X_stat], y, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.2,
    verbose=1
)

# 保存模型
model.save('st_sdm_model.h5')
print("模型训练完成并保存")

# 可视化训练过程
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失') 
plt.title('模型训练过程')
plt.xlabel('Epochs')
plt.ylabel('损失')
plt.legend()
plt.savefig('training_history.png')
plt.show()