import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation
import rasterio
import os
from sklearn.preprocessing import MinMaxScaler

def load_raster_data(file_paths, common_mask=None):
    """
    加载多个栅格文件并返回合并的数据矩阵
    
    :param file_paths: 栅格文件路径列表
    :param common_mask: 共同的有效区域掩膜(可选)
    :return: 数据矩阵 (n_samples, n_features) 和有效区域掩膜
    """
    raster_data = []
    shapes = []
    mask = None
    
    # 读取所有栅格
    for path in file_paths:
        with rasterio.open(path) as src:
            data = src.read(1)  # 读取第一个波段
            shapes.append(data.shape)
            
            # 创建该栅格的有效数据掩码
            current_mask = (~np.isnan(data)) & (data != 0)
            
            # 如果是第一个栅格，初始化mask
            if mask is None:
                mask = current_mask
            else:
                # 取交集，确保所有栅格在该位置都有有效数据
                mask = mask & current_mask
    
    # 再次读取数据，但这次只提取共同有效区域的值
    raster_data = []
    for path in file_paths:
        with rasterio.open(path) as src:
            data = src.read(1)
            valid_data = data[mask]
            raster_data.append(valid_data)
            
    # 检查所有栅格形状是否一致        
    if len(set([s for s in shapes])) > 1:
        raise ValueError(f"栅格尺寸不一致: {shapes}")
        
    # 将栅格数据合并为矩阵形式 [n_samples, n_features]
    X = np.column_stack(raster_data)
    
    return X, mask

def normalize_data(X):
    """
    标准化数据矩阵
    
    :param X: 原始数据矩阵 (n_samples, n_features)
    :return: 标准化后的数据矩阵
    """
    X_norm = np.zeros_like(X, dtype=float)
    
    # 对每列(每个指标)分别进行最小-最大标准化
    for i in range(X.shape[1]):
        col = X[:, i]
        # 排除无效值
        valid = ~np.isnan(col)
        if np.sum(valid) > 0:  # 确保有有效值
            min_val = np.min(col[valid])
            max_val = np.max(col[valid])
            # 避免除以零
            if max_val > min_val:
                X_norm[valid, i] = (col[valid] - min_val) / (max_val - min_val)
            else:
                X_norm[valid, i] = 0.5  # 如果所有值相同，设置为0.5
    
    return X_norm

def entropy_weights(X):
    """按照指定公式实现的熵权法"""
    eps = 1e-10
    
    # 归一化处理
    p = (X - np.min(X, axis=0)) / (np.ptp(X, axis=0) + eps)
    p = (p + eps) / (p.sum(axis=0) + eps * X.shape[0])  # 防止除零
    
    # 计算熵值
    e = -np.sum(p * np.log(p + eps), axis=0) / np.log(X.shape[0])
    
    # 计算权重
    d = 1 - e
    return d / (d.sum() + eps)

def quick_sensitivity_test(X, noise_level=0.05, n_iter=100):
    """
    快速敏感性测试，评估指标权重对噪声的敏感性
    
    :param X: 标准化后的数据矩阵 (n_samples, n_features)
    :param noise_level: 噪声水平 (默认0.05)
    :param n_iter: 迭代次数 (默认100)
    :return: 统计结果和图表对象
    """
    n_samples, n_features = X.shape
    weights_list = []
    base_weights = entropy_weights(X)  # 原始权重
    
    for _ in range(n_iter):
        # 添加噪声 - 对于归一化数据(0-1范围)的特殊处理
        # 使用乘性噪声而非加性噪声，确保数据保持正值
        noise = np.random.uniform(1-noise_level, 1+noise_level, X.shape)
        X_noisy = X * noise
        
        # 裁剪到0-1范围
        X_noisy = np.clip(X_noisy, 0, 1)
        
        # 计算权重
        weights = entropy_weights(X_noisy)
        weights_list.append(weights)
    
    weights_array = np.array(weights_list)
    
    # 计算统计信息
    mean_weights = np.mean(weights_array, axis=0)
    std_weights = np.std(weights_array, axis=0)
    max_change = np.max(weights_array, axis=0) - np.min(weights_array, axis=0)
    cv_weights = variation(weights_array, axis=0)
    
    stats = {
        'mean': mean_weights,
        'std': std_weights,
        'max_change': max_change,
        'cv': cv_weights,
        'base_weights': base_weights,
        'all_weights': weights_array
    }
    
    # 绘制图表 - 增强版本
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左侧图 - 权重分布
    ax1.errorbar(range(n_features), mean_weights, yerr=std_weights, fmt='o', capsize=5)
    ax1.plot(range(n_features), base_weights, 'rs', label='Original Weights')
    ax1.set_title('Weight sensity test')
    ax1.set_xlabel('Indicators')
    ax1.set_ylabel('Weight Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右侧图 - 权重变化箱形图
    ax2.boxplot(weights_array)
    ax2.plot(range(1, n_features+1), base_weights, 'rs', label='Original Weights')
    ax2.set_title('Weight Distribution')
    ax2.set_xlabel('Indicators')
    ax2.set_ylabel('Weight Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    return stats, fig

# ============= 主程序使用实际栅格数据 =============
if __name__ == "__main__":
    # 指定三个栅格文件路径
    raster_paths = [
        r"D:\code\house\PPI_output\PPI_sz_clipped_raster_2017_06.tif",  # PPI指标
        r"D:\code\house\hsi_results_utm50n.tif",  # HSI指标
        r"D:\code\house\poi_hhi_output.tif"   # HHI指标
    ]
    
    # 检查文件是否存在
    missing_files = [path for path in raster_paths if not os.path.exists(path)]
    if missing_files:
        print(f"错误: 以下文件不存在:")
        for path in missing_files:
            print(f"  - {path}")
        exit(1)
    
    # 加载栅格数据
    print("加载栅格数据...")
    try:
        X_raw, mask = load_raster_data(raster_paths)
        print(f"数据加载成功: {X_raw.shape[0]} 个有效像素, {X_raw.shape[1]} 个指标")
        
        # 显示原始数据的基本统计信息
        print("\n原始指标统计:")
        for i in range(X_raw.shape[1]):
            mean = np.nanmean(X_raw[:, i])
            std = np.nanstd(X_raw[:, i])
            min_val = np.nanmin(X_raw[:, i])
            max_val = np.nanmax(X_raw[:, i])
            print(f"指标 {i+1}: 均值={mean:.3f}, 标准差={std:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")
        
        # 标准化数据
        print("\n执行数据标准化...")
        X = normalize_data(X_raw)
        
        # 显示标准化后的统计信息
        print("\n标准化后指标统计:")
        for i in range(X.shape[1]):
            mean = np.nanmean(X[:, i])
            std = np.nanstd(X[:, i])
            min_val = np.nanmin(X[:, i])
            max_val = np.nanmax(X[:, i])
            print(f"指标 {i+1}: 均值={mean:.3f}, 标准差={std:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]")
        
        # 计算熵权法权重
        base_weights = entropy_weights(X)
        print("\n熵权法计算结果:")
        feature_names = ['PPI', 'HSI', 'HHI']
        for i, (name, weight) in enumerate(zip(feature_names, base_weights)):
            print(f"{name}: {weight:.4f}")
        
        # 执行敏感性测试
        print("\n执行敏感性测试...")
        stats, fig = quick_sensitivity_test(X, noise_level=0.05, n_iter=1000)
        
        # 添加标签
        plt.figure(fig.number)
        axes = fig.get_axes()
        for ax in axes:
            ax.set_xticks(range(len(feature_names)) if ax.get_xticks().size != len(feature_names) 
                         else range(1, len(feature_names)+1))
            ax.set_xticklabels(feature_names)
        
        # 输出统计结果
        print("\n=== 敏感性测试报告 ===")
        print(f"原始权重: {base_weights.round(4)}")
        print(f"平均权重: {stats['mean'].round(4)}")
        print(f"标准差: {stats['std'].round(4)}")
        print(f"最大变化: {stats['max_change'].round(4)}")
        print(f"变异系数: {stats['cv'].round(4)}")
        
        # 保存图表
        fig.savefig("weights_sensitivity.png", dpi=300, bbox_inches="tight")
        print("图表已保存至 weights_sensitivity.png")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        import traceback
        print(f"错误: {str(e)}")
        print(traceback.format_exc())