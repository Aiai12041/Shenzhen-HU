"""
人口压力指数(PPI)计算模块
该模块基于时空数据计算区域人口压力指数
"""
from dask.distributed import Client
import os
import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_tif_to_xarray(path):
    """
    加载GeoTIFF文件到xarray数据结构
    
    Args:
        path: GeoTIFF文件路径
        
    Returns:
        ds: 包含时空数据的xarray DataArray
        meta: 栅格元数据
    """
    with rasterio.open(path) as src:
        data = src.read()  # Shape: (96, H, W)
        meta = src.meta
    
    # 获取空间尺寸
    H, W = data.shape[1], data.shape[2]
    
    # 重塑数据为(4, 24, H, W)的形状
    reshaped_data = data.reshape(4, 24, H, W)
    
    # 定义时间聚类和小时
    time_clusters = ['workday_in', 'workday_out', 'holiday_in', 'holiday_out']
    hours = list(range(24))
    
    # 直接创建带有time_cluster和hour维度的DataArray
    ds = xr.DataArray(
        data=reshaped_data,
        dims=('time_cluster', 'hour', 'y', 'x'),
        coords={
            'time_cluster': time_clusters,
            'hour': hours
        }
    )
    
    return ds, meta

def extract_features(ds):
    """
    从数据中提取特征
    
    Args:
        ds: xarray DataArray
        
    Returns:
        features: 包含提取特征的xarray Dataset
    """
    # 时段划分
    peak_hours = {'morning': [7,8,9], 'evening': [17,18,19]}
    night_hours = [22,23,0,1,2,3]
    
    features = xr.Dataset()
    # 工作日流入通勤压力
    features['workday_in_peak'] = ds.sel(time_cluster='workday_in', hour=peak_hours['morning']).mean(dim='hour')
    # 节假日夜间活跃度
    features['holiday_night'] = ds.sel(time_cluster='holiday_in', hour=night_hours).sum(dim='hour')
    # 流出/流入比
    features['out_in_ratio'] = (ds.sel(time_cluster='workday_out').sum() + 1e-6) / (ds.sel(time_cluster='workday_in').sum() + 1e-6)
    return features


def dynamic_weights(features):
    """
    使用PCA计算特征的动态权重
    
    Args:
        features: 特征Dataset
        
    Returns:
        weights: 权重向量
    """
    # 将Dataset转换为DataArray
    features_array = features.to_array()
    
    # 检查实际维度名称
    variable_dim = 'variable'  # xarray默认使用'variable'作为转换后的变量维度名
    
    # 将数据重塑为二维数组：(空间点数, 特征数)
    # 即将y和x维度展平为样本维度，variable维度作为特征维度
    flat_data = features_array.stack(sample=('y', 'x')).transpose(variable_dim, 'sample').values.T
    
    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(flat_data)
    
    # PCA提取主成分
    pca = PCA(n_components=min(3, scaled_data.shape[1]))
    components = pca.fit_transform(scaled_data)
    
    # 权重为各主成分方差解释比例
    weights = pca.explained_variance_ratio_
    return weights


def save_geotiff(data, output_path, meta):
    """
    将数据保存为GeoTIFF
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
        meta: 栅格元数据
    """
    # 复制元数据并更新
    meta_copy = meta.copy()
    meta_copy.update({
        'count': 1,
        'dtype': 'float32'
    })
    
    with rasterio.open(output_path, 'w', **meta_copy) as dst:
        dst.write(data.values.astype('float32'), 1)


def process_all_months(input_dir, output_dir):
    """
    批量处理文件夹下所有TIF文件
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 启用Dask并行
    client = Client()
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            path = os.path.join(input_dir, filename)
            ds, meta = load_tif_to_xarray(path)
            features = extract_features(ds)
            
            # 计算动态权重
            weights = dynamic_weights(features)
            
            # 合成PPI
            ppi = (weights[0] * features['workday_in_peak'] +
                  weights[1] * features['holiday_night'] +
                  weights[2] * features['out_in_ratio'])
            
            # 保存结果
            save_geotiff(ppi, os.path.join(output_dir, f"PPI_{filename}"), meta)
    
    # 关闭客户端
    client.close()


def plot_spatiotemporal(ppi_data, timestamp):
    """
    绘制PPI的空间分布和时间趋势
    
    Args:
        ppi_data: 包含PPI数据的xarray DataArray
        timestamp: 要显示的时间戳
    """
    fig, axes = plt.subplots(1, 2, figsize=(20,8))
    
    # 空间分布
    ppi_data.sel(time=timestamp).plot(ax=axes[0], cmap='viridis', 
                                     cbar_kwargs={'label': 'PPI'})
    axes[0].set_title(f"Spatial Distribution - {timestamp}")
    
    # 时间序列
    ppi_data.mean(dim=['x','y']).plot(ax=axes[1], marker='o')
    axes[1].set_title("Temporal Trend of Regional Average PPI")
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 示例用法
    input_dir = "D:\houseuse\clipped_od"
    output_dir = "./PPI_output"
    
    # 处理所有月份数据
    process_all_months(input_dir, output_dir)
    
    # 可视化示例（需要适配实际数据结构）
    # ppi_file = os.path.join(output_dir, "PPI_example.tif")
    # with rasterio.open(ppi_file) as src:
    #     ppi_data = xr.DataArray(src.read(1), dims=('y', 'x'))
    # plot_spatiotemporal(ppi_data, "2025-03")
    # plt.savefig("ppi_visualization.png")
    # plt.show()