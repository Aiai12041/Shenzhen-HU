import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from datetime import datetime
import glob
import re

class Config:
    # 路径配置
    ppi_dir = "./PPI_output/"
    hsi_path = "./hsi_results_utm50n.tif"
    hhi_path = "./poi_hhi_output.tif"
    output_dir = "./hcewi/"
    ppi_pattern = r"PPI_sz_clipped_raster_(\d{4})_(\d{2})\.tif$"  # PPI文件名匹配模式

    # 空间参考配置（以HSI文件为基准）
    target_crs = None  # 自动获取
    target_transform = None
    target_shape = None

def setup_output_metadata():
    """设置输出空间参考元数据"""
    with rasterio.open(Config.hsi_path) as src:
        Config.target_crs = src.crs
        Config.target_transform = src.transform
        Config.target_shape = src.shape

def parse_ppi_datetime(filename):
    """使用正则表达式解析时间信息"""
    match = re.search(Config.ppi_pattern, filename, re.IGNORECASE)
    if match:
        year, month = map(int, match.groups())
        return datetime(year, month, 1)
    raise ValueError(f"无法解析文件名中的时间信息：{filename}")

def load_static_data(path):
    """加载静态数据并重投影到目标坐标系"""
    with rasterio.open(path) as src:
        # 如果坐标系不一致则自动重投影
        if src.crs != Config.target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, Config.target_crs, src.width, src.height, *src.bounds)
            data = np.empty((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=Config.target_crs)
            return data
        return src.read(1)

def process_ppi_files():
    """改进的文件处理函数"""
    ppi_files = sorted(
        glob.glob(os.path.join(Config.ppi_dir, "PPI_sz_clipped_raster_*.tif")),
        key=lambda x: parse_ppi_datetime(os.path.basename(x))
    )
    
    for ppi_file in ppi_files:
        # 解析时间并过滤
        date = parse_ppi_datetime(os.path.basename(ppi_file))
        if date < datetime(2017,6,1) or date > datetime(2019,6,30):
            continue
        
        # 显示处理进度
        print(f"正在处理：{os.path.basename(ppi_file)}")
        
        # 加载和处理数据
        with rasterio.open(ppi_file) as src:
            ppi = src.read(1)
            # 自动重投影对齐
            if src.crs != Config.target_crs:
                ppi = reproject_to_target(ppi, src)
        
        # 执行计算流程
        calculate_hcewi_for_month(ppi, date)

def reproject_to_target(data, src):
    """数据重投影到目标坐标系"""
    transformed = np.empty(Config.target_shape, dtype=np.float32)
    reproject(
        source=data,
        destination=transformed,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=Config.target_transform,
        dst_crs=Config.target_crs)
    return transformed

def safe_normalization(data, name=""):
    """安全归一化到0-1范围"""
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    denominator = data_max - data_min
    
    if denominator < 1e-10:
        print(f"警告：{name}数据值域接近零({data_min:.2f}-{data_max:.2f})，使用默认归一化")
        normalized = np.zeros_like(data)
    else:
        normalized = (data - data_min) / denominator
        
    # 处理NaN值
    normalized = np.nan_to_num(normalized, nan=0.0)
    return np.clip(normalized, 0, 1)

def calculate_hcewi_for_month(ppi, date):
    """单个月份的HCEWI计算"""
    # 检查数据有效性
    if np.all(np.isnan(ppi)):
        print(f"警告：{date.strftime('%Y%m')} 的PPI数据全为NaN，跳过处理")
        return
    
    # 数据标准化（新添加的0-1归一化）
    ppi_norm = safe_normalization(ppi, name="PPI")
    
    # 加载静态数据（仅首次加载时处理）
    if not hasattr(Config, 'hsi_norm'):
        print("加载静态数据...")
        
        # 加载并标准化HSI
        Config.hsi = load_static_data(Config.hsi_path)
        Config.hsi_norm = safe_normalization(Config.hsi, name="HSI")
        
        # 加载并标准化HHI
        Config.hhi = load_static_data(Config.hhi_path)
        Config.hhi_norm = safe_normalization(Config.hhi, name="HHI")
        
        # 计算空间权重矩阵
        Config.weights = calculate_spatial_weights(Config.hsi_norm, Config.hhi_norm)
    
    # 合成指标（添加安全约束）
    hcewi = (
        Config.weights[...,0] * ppi_norm +
        Config.weights[...,1] * Config.hsi_norm +
        Config.weights[...,2] * Config.hhi_norm
    )
    
    # 结果后处理
    hcewi = np.clip(hcewi, 0, 1)  # 强制限定到0-1范围
    hcewi = np.nan_to_num(hcewi, nan=0.0)  # 清除残留NaN
    
    # 输出诊断信息
    valid_ratio = np.mean((hcewi > 0) & (hcewi < 1)) * 100
    print(f"[{date.strftime('%Y%m')}] 有效值比例：{valid_ratio:.1f}% 数值范围：({hcewi.min():.3f}, {hcewi.max():.3f})")
    
    # 输出结果
    output_path = os.path.join(Config.output_dir, f"hcewi_{date.strftime('%Y%m')}.tif")
    write_geotiff(output_path, hcewi)

def calculate_spatial_weights(hsi, hhi):
    """改进的空间权重计算方法"""
    valid_mask = ~np.isnan(hsi) & ~np.isnan(hhi)
    
    # 更新变异系数计算方法
    def safe_cv(data):
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if mean < 1e-10:  # 处理接近零的均值
            return 0.0
        return std / mean
    
    window_size = 7
    weights = np.zeros((*hsi.shape, 3))
    
    for i in range(hsi.shape[0]):
        for j in range(hsi.shape[1]):
            if not valid_mask[i,j]:
                weights[i,j] = [0.333, 0.333, 0.334]
                continue
                
            i_min = max(0, i - window_size//2)
            i_max = min(hsi.shape[0], i + window_size//2 + 1)
            j_min = max(0, j - window_size//2)
            j_max = min(hsi.shape[1], j + window_size//2 + 1)
            
            local_hsi = hsi[i_min:i_max, j_min:j_max]
            local_hhi = hhi[i_min:i_max, j_min:j_max]
            
            # 添加平滑处理
            w1 = safe_cv(local_hsi) if local_hsi.size > 0 else 0.0
            w2 = safe_cv(local_hhi) if local_hhi.size > 0 else 0.0
            w3 = 1.0  # 基础权重
            
            # 权重归一化
            total = w1 + w2 + w3
            if total < 1e-10:
                weights[i,j] = [0.333, 0.333, 0.334]
            else:
                weights[i,j] = [w1/total, w2/total, w3/total]
    
    return weights

def write_geotiff(path, data):
    """写入GeoTIFF文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=np.float32,
        crs=Config.target_crs,
        transform=Config.target_transform,
        nodata=np.nan
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"生成文件：{path}")

if __name__ == "__main__":
    setup_output_metadata()
    process_ppi_files()
    print("处理完成！")
