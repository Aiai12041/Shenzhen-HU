import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib import colors
import libpysal as lps
from esda.moran import Moran, Moran_Local

def load_hcewi_files(folder_path, pattern=r"HCEWI_.*?_(\d{4})_(\d{2})\.tif"):
    """
    加载文件夹中的所有HCEWI结果文件
    
    :param folder_path: HCEWI文件所在的文件夹路径
    :param pattern: 用于提取日期的正则表达式模式
    :return: 字典 {日期字符串: (数据数组, 掩码)}
    """
    results = {}
    
    # 列出所有文件
    files = [f for f in os.listdir(folder_path) if f.startswith("HCEWI_") and f.endswith(".tif")]
    files.sort()  # 按名称排序
    
    if not files:
        print(f"警告: 在 {folder_path} 中未找到HCEWI文件")
        return results
        
    for filename in files:
        # 提取日期
        match = re.search(pattern, filename)
        if match:
            year, month = match.groups()
            date_str = f"{year}{month}"
            
            # 读取文件
            file_path = os.path.join(folder_path, filename)
            with rasterio.open(file_path) as src:
                data = src.read(1)
                # 创建有效数据掩码 (非NaN且非零)
                mask = (~np.isnan(data)) & (data != 0)
                
                results[date_str] = (data, mask)
            print(f"已加载 {date_str} 的HCEWI数据，有效像素: {np.sum(mask)}")
    
    print(f"总共加载了 {len(results)} 个月份的HCEWI数据")
    return results

def run_lisa_analysis(data, mask, p_threshold=0.05):
    """
    执行LISA局部空间自相关分析
    
    :param data: HCEWI数据数组
    :param mask: 有效区域掩码
    :param p_threshold: 显著性阈值
    :return: 分析结果字典
    """
    from esda.moran import Moran_Local
    
    # 创建带掩码的数组
    masked_array = np.full_like(data, np.nan, dtype=float)
    masked_array[mask] = data[mask]
    
    # 将NaN替换为均值
    mean_value = np.nanmean(masked_array)
    filled_array = np.nan_to_num(masked_array, nan=mean_value)
    
    # 生成空间权重矩阵
    w = lps.weights.lat2W(data.shape[0], data.shape[1], rook=False)
    w.transform = 'r'  # 行标准化
    
    # 计算全局Moran's I
    moran_global = Moran(filled_array.flatten(), w, permutations=999)
    
    # 计算局部Moran's I
    lisa = Moran_Local(filled_array.flatten(), w, permutations=999)
    
    # 处理结果，创建聚类地图
    sig = lisa.p_sim < p_threshold
    spots = np.zeros(lisa.y.shape, dtype=int)
    spots[(sig) & (lisa.q == 1)] = 1  # 高-高
    spots[(sig) & (lisa.q == 3)] = 3  # 低-低
    spots[(sig) & (lisa.q == 2)] = 2  # 低-高
    spots[(sig) & (lisa.q == 4)] = 4  # 高-低
    
    # 统计各类型数量
    counts = {
        'HH': np.sum(spots == 1),
        'LH': np.sum(spots == 2),
        'LL': np.sum(spots == 3),
        'HL': np.sum(spots == 4),
        'Not Significant': np.sum(spots == 0)
    }
    
    # 返回结果
    return {
        'global_moran': {
            'I': moran_global.I,
            'p_value': moran_global.p_norm,
            'z_score': moran_global.z_norm
        },
        'lisa': lisa,
        'cluster_map': spots.reshape(data.shape),
        'counts': counts
    }

def save_lisa_cluster_as_tif(cluster_map, output_path, reference_tif):
    """
    将LISA聚类结果保存为GeoTIFF栅格文件
    
    :param cluster_map: 聚类结果数组
    :param output_path: 输出文件路径
    :param reference_tif: 参考TIF文件路径(用于获取地理参考信息)
    """
    # 读取参考文件的元数据
    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
    
    # 更新元数据
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 0
    })
    
    # 保存聚类结果
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(cluster_map.astype('uint8'), 1)
    
    print(f"已保存LISA聚类栅格: {output_path}")
    
def analyze_hcewi_timeseries(folder_path, output_folder=None):
    """
    分析HCEWI结果的时间序列变化，并将LISA聚类结果保存为栅格
    
    :param folder_path: HCEWI结果文件夹
    :param output_folder: LISA结果输出文件夹，默认为folder_path下的lisa_results子文件夹
    """
    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(folder_path, "lisa_results")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载所有HCEWI文件
    hcewi_data = load_hcewi_files(folder_path)
    
    if not hcewi_data:
        print("未找到HCEWI数据文件，分析终止")
        return
    
    # 存储分析结果
    results = {}
    global_moran_values = []
    dates = []
    cluster_counts = []
    
    # 对每个月份执行LISA分析
    for date_str, (data, mask) in hcewi_data.items():
        print(f"\n分析 {date_str} 的HCEWI数据...")
        
        # 获取原始文件路径(用于提取地理参考信息)
        original_file = None
        for file in os.listdir(folder_path):
            # 提取文件名中的年份和月份
            year_month_match = re.search(r"(\d{4})_(\d{2})", file)
            
            if year_month_match:
                file_year, file_month = year_month_match.groups()
                file_date_str = f"{file_year}{file_month}"  # e.g. "201707"
                
                # 检查是否与当前处理的日期匹配
                if file_date_str == date_str and file.startswith("HCEWI_"):
                    original_file = os.path.join(folder_path, file)
                    break
        
        if original_file is None:
            print(f"警告：无法找到 {date_str} 的原始文件，跳过保存栅格")
            continue
        
        # 执行LISA分析
        lisa_result = run_lisa_analysis(data, mask)
        
        # 保存LISA聚类结果为栅格文件
        lisa_output_path = os.path.join(output_folder, f"LISA_cluster_{date_str}.tif")
        save_lisa_cluster_as_tif(lisa_result['cluster_map'], lisa_output_path, original_file)
        
        # 存储结果
        results[date_str] = lisa_result
        dates.append(date_str)
        global_moran_values.append(lisa_result['global_moran']['I'])
        cluster_counts.append(lisa_result['counts'])
        
        # 输出全局Moran's I结果
        print(f"全局Moran's I: {lisa_result['global_moran']['I']:.4f} (p值: {lisa_result['global_moran']['p_value']:.4f})")
        print("LISA聚类统计:")
        for cluster_type, count in lisa_result['counts'].items():
            print(f"  - {cluster_type}: {count} 个像素")
    
    # 可视化全局Moran's I时间序列
    plt.figure(figsize=(10, 6))
    plt.plot(dates, global_moran_values, 'o-', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.title("The temporal variation of HCEWI global spatial autocorrelation")
    plt.xlabel("Date")
    plt.ylabel("Moran's I")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "global_moran_timeseries.png"))
    
    # 可视化聚类类型变化
    cluster_types = ['HH', 'LH', 'LL', 'HL']
    cluster_data = pd.DataFrame({
        'Date': dates * len(cluster_types),
        'ClusterType': sum([[ct] * len(dates) for ct in cluster_types], []),
        'Count': sum([[cluster_counts[i][ct] for i in range(len(dates))] for ct in cluster_types], [])
    })
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=cluster_data, x='Date', y='Count', hue='ClusterType', marker='o')
    plt.title("The temporal variation of HCEWI local spatial clustering")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "cluster_counts_timeseries.png"))
    
    # 移除图像比较部分，改为保存每个月份的聚类结果栅格
    
    return results

# 执行分析
if __name__ == "__main__":
    # 指定HCEWI结果所在的文件夹
    hcewi_folder = "./hcewi"  # 根据实际情况修改
    lisa_output_folder = "./lisa_results"  # 输出文件夹
    
    # 检查文件夹是否存在
    if not os.path.exists(hcewi_folder):
        print(f"错误: 文件夹 {hcewi_folder} 不存在")
    else:
        # 运行时序分析
        results = analyze_hcewi_timeseries(hcewi_folder, lisa_output_folder)
    
        # 显示所有图表
        plt.show()