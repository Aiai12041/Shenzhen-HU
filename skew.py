import os
import glob
import numpy as np
import rasterio
from scipy import stats
from tqdm import tqdm  # 可视化进度条


def calculate_skewness_from_rasters(input_folder, output_path, pattern="*.tif"):
    """
    计算文件夹中所有栅格文件同一位置像素的偏度系数
    
    Args:
        input_folder: 输入栅格文件夹路径
        output_path: 输出偏度系数栅格路径
        pattern: 文件匹配模式，默认为所有tif文件
    """
    # 查找所有匹配的栅格文件
    raster_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    
    if not raster_files:
        raise ValueError(f"在 {input_folder} 中没有找到匹配的栅格文件")
    
    print(f"找到 {len(raster_files)} 个栅格文件")
    
    # 从第一个文件获取元数据和尺寸信息
    with rasterio.open(raster_files[0]) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
    
    # 创建一个数组来存储所有栅格数据
    all_data = np.zeros((len(raster_files), height, width), dtype=np.float32)
    valid_count = np.zeros((height, width), dtype=np.int16)
    
    # 读取所有栅格数据
    print("正在读取栅格数据...")
    for i, raster_file in enumerate(tqdm(raster_files)):
        with rasterio.open(raster_file) as src:
            # 检查尺寸是否一致
            if src.height != height or src.width != width:
                raise ValueError(f"栅格 {raster_file} 的尺寸与第一个栅格不一致")
            
            data = src.read(1)  # 读取第一个波段
            all_data[i] = data
            valid_count += (~np.isnan(data) & (data != src.nodata if src.nodata is not None else True)).astype(np.int16)
    
    # 计算偏度系数
    print("计算偏度系数...")
    # 初始化结果数组
    skewness = np.zeros((height, width), dtype=np.float32)
    skewness.fill(np.nan)  # 默认值为NaN
    
    # 需要至少3个有效值才能计算偏度
    min_valid = 3
    
    # 对每个像素位置计算偏度
    for i in range(height):
        for j in range(width):
            pixel_values = all_data[:, i, j]
            # 过滤无效值
            valid_values = pixel_values[~np.isnan(pixel_values)]
            if len(valid_values) >= min_valid:
                skewness[i, j] = stats.skew(valid_values)
    
    # 另一种实现方式 - 向量化计算（可能更快但内存消耗大）
    # 重塑数组以便进行向量化计算
    # reshaped_data = all_data.reshape(len(raster_files), -1)
    # skewness_flat = np.zeros(height * width, dtype=np.float32)
    # skewness_flat.fill(np.nan)
    # valid_pixels = valid_count.flatten() >= min_valid
    # for i in range(reshaped_data.shape[1]):
    #     if valid_pixels[i]:
    #         values = reshaped_data[:, i]
    #         valid_values = values[~np.isnan(values)]
    #         if len(valid_values) >= min_valid:
    #             skewness_flat[i] = stats.skew(valid_values)
    # skewness = skewness_flat.reshape(height, width)
    
    # 更新元数据
    meta.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': np.nan
    })
    
    # 保存结果
    print(f"保存结果到 {output_path}...")
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(skewness.astype(np.float32), 1)
    
    # 输出基本统计信息
    valid_skew = skewness[~np.isnan(skewness)]
    if len(valid_skew) > 0:
        print("\n偏度系数统计信息:")
        print(f"最小值: {np.min(valid_skew):.4f}")
        print(f"最大值: {np.max(valid_skew):.4f}")
        print(f"平均值: {np.mean(valid_skew):.4f}")
        print(f"中位数: {np.median(valid_skew):.4f}")
        print(f"正偏度占比: {np.sum(valid_skew > 0) / len(valid_skew) * 100:.2f}%")
        print(f"负偏度占比: {np.sum(valid_skew < 0) / len(valid_skew) * 100:.2f}%")
    else:
        print("警告: 没有有效的偏度系数值")
    
    return skewness


if __name__ == "__main__":
    # 设置输入输出路径
    input_folder = r"D:\code\house\hcewi"  # 修改为您的输入文件夹
    output_path = r"D:\code\house\skewness_result.tif"  # 修改为您的输出文件路径
    
    # 调用函数计算偏度系数
    result = calculate_skewness_from_rasters(input_folder, output_path)
    
    print("处理完成!")