import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import boxcox
import warnings
import os

class HCEProcessor:
    def __init__(self, hsi_path, hhi_path):
        # 初始化基础数据
        self.hsi, self.hsi_meta = self._load_base_data(hsi_path)
        self.hhi = self._load_base_data(hhi_path)[0]
        
        # 生成综合掩膜（非空、非零、双数据有效区域）
        self.mask = self._create_compound_mask(self.hsi, self.hhi)
        
        # 数据预处理（保留原始值）
        self.hsi_inv = 1 - self._normalize_with_mask(self.hsi)
        self.hhi_non = np.power(self._normalize_with_mask(self.hhi), 1.5)

    def _load_base_data(self, path):
        """加载基础数据并保留原始值"""
        with rasterio.open(path) as src:
            data = src.read(1)
            return data, src.profile

    def _create_compound_mask(self, hsi, hhi):
        """创建复合掩膜：非空、非零、双数据有效区域"""
        valid_hsi = (~np.isnan(hsi)) & (hsi != 0)
        valid_hhi = (~np.isnan(hhi)) & (hhi != 0)
        return valid_hsi & valid_hhi

    def _normalize_with_mask(self, data):
        """基于复合掩膜的归一化"""
        valid_data = data[self.mask]
        if valid_data.size == 0:
            return np.zeros_like(data)
            
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val - min_val < 1e-10:
            return np.zeros_like(data)
            
        normalized = (data - min_val) / (max_val - min_val)
        return np.where(self.mask, normalized, np.nan)

    def _process_ppi(self, ppi_path):
        """处理PPI数据时应用复合掩膜"""
        ppi = self._reproject(ppi_path) if self._needs_reproject(ppi_path) else self._load_ppi(ppi_path)
        return self._safe_boxcox(np.where(self.mask, ppi, np.nan))

    def process_month(self, ppi_path, output_dir):
        """带复合掩膜的综合计算"""
        ppi_norm = self._process_ppi(ppi_path)
        
        # 构建有效数据矩阵（仅掩膜区域）
        valid_mask = self.mask.flatten()
        X = np.stack([
            ppi_norm.flatten()[valid_mask],
            self.hsi_inv.flatten()[valid_mask],
            self.hhi_non.flatten()[valid_mask]
        ], axis=1)

        # 熵权法计算（仅有效数据）
        if X.size == 0:
            print(f"Warning: No valid data for {ppi_path}")
            return
            
        weights = self._entropy_weights(X)
        
        # 合成指标计算
        with np.errstate(invalid='ignore'):
            hcewi = (
                weights[0] * ppi_norm +
                weights[1] * self.hsi_inv +
                weights[2] * self.hhi_non
            )
        
        # 后处理：保留原始无效区域
        final_output = np.where(self.mask, np.clip(hcewi, 0, 1), 0.0)
        self._save_results(final_output, ppi_path, output_dir)
    
    def _needs_reproject(self, ppi_path):
        """判断是否需要重投影"""
        with rasterio.open(ppi_path) as src:
            return (src.crs != self.hsi_meta['crs']) or \
                   (src.transform != self.hsi_meta['transform']) or \
                   (src.width != self.hsi_meta['width']) or \
                   (src.height != self.hsi_meta['height'])

    def _reproject(self, ppi_path):
        """执行重投影到基准坐标系"""
        with rasterio.open(ppi_path) as src:
            # 创建目标数组
            ppi_data = np.empty((self.hsi_meta['height'], self.hsi_meta['width']), 
                              dtype=np.float32)
            
            # 执行重投影
            reproject(
                source=rasterio.band(src, 1),
                destination=ppi_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.hsi_meta['transform'],
                dst_crs=self.hsi_meta['crs'],
                resampling=Resampling.bilinear
            )
            return ppi_data

    def _load_ppi(self, ppi_path):
        """直接加载PPI数据"""
        with rasterio.open(ppi_path) as src:
            return src.read(1)
        
    def _safe_boxcox(self, data):
        """带异常处理的Box-Cox变换"""
        valid_data = data[self.mask]
        if valid_data.size == 0 or np.all(valid_data == valid_data[0]):
            return np.zeros_like(data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 确保数据正值
            data_shifted = valid_data - np.nanmin(valid_data) + 1e-10
            try:
                transformed, _ = boxcox(data_shifted)
            except ValueError as e:
                print(f"Box-Cox warning: {str(e)}, using log transform")
                transformed = np.log(data_shifted + 1e-10)

        # 归一化处理
        normalized = (transformed - np.min(transformed)) / (np.ptp(transformed) + 1e-10)
        output = np.full_like(data, np.nan)
        output[self.mask] = normalized
        return np.nan_to_num(output, nan=0.0)

    def _entropy_weights(self, X):
        """带鲁棒性处理的熵权法"""
        eps = 1e-10
        # 归一化处理
        p = (X - np.min(X, axis=0)) / (np.ptp(X, axis=0) + eps)
        p = (p + eps) / (p.sum(axis=0) + eps * X.shape[0])  # 防止除零
        e = -np.sum(p * np.log(p + eps), axis=0) / np.log(X.shape[0])
        d = 1 - e
        return d / (d.sum() + eps)

    def _save_results(self, data, ppi_path, output_dir):
        """增强的保存方法"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"HCEWI_{os.path.basename(ppi_path)}")
        
        with rasterio.open(output_path, 'w', 
                         driver=self.hsi_meta['driver'],
                         height=self.hsi_meta['height'],
                         width=self.hsi_meta['width'],
                         count=1,
                         dtype=np.float32,
                         crs=self.hsi_meta['crs'],
                         transform=self.hsi_meta['transform']) as dst:
            dst.write(data.astype(np.float32), 1)
            dst.nodata = 0.0


if __name__ == "__main__":
    # 配置路径
    processor = HCEProcessor(
        hsi_path="./hsi_results_utm50n.tif",
        hhi_path="./poi_hhi_output.tif"
    )
    
    # 处理所有PPI文件
    ppi_files = [
        os.path.join("./PPI_output", f) 
        for f in os.listdir("./PPI_output") 
        if f.startswith("PPI_") and f.endswith(".tif")
    ]
    
    for ppi_file in ppi_files:
        print(f"Processing: {os.path.basename(ppi_file)}")
        processor.process_month(
            ppi_path=ppi_file,
            output_dir="./HCEWI"
        )
