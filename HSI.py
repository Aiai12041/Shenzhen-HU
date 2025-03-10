import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from rasterio.features import rasterize
from shapely.geometry import Point, box
from pyproj import Transformer

def coordinate_transformation():
    """坐标转换与数据整合流程"""
    # 配置参数
    config = {
        'poi_csv': r"D:\houseuse\yga_poi_file_84\深圳市_poi_84.csv",
        'building_raster': r"D:\houseuse\generated_vbd_sz.tif",
        'output_hsi': r"D:\houseuse\hsi_results_utm50n.tif"  # 修改为.tif后缀
    }

    # 1. 读取建筑密度栅格元数据
    with rasterio.open(config['building_raster']) as src:
        raster_crs = src.crs  # 应为EPSG:32650
        raster_transform = src.transform
        raster_bounds = src.bounds
        raster_shape = src.shape
        raster_meta = src.meta.copy()
        print(f"栅格坐标系: {raster_crs}\n范围: {raster_bounds}")

    # 2. 读取并转换POI数据坐标系
    poi_df = pd.read_csv(config['poi_csv'], encoding='gbk')
    
    # 2.1 根据类型前四个字符分类住宅POI
    poi_df['is_residential'] = poi_df['类型'].apply(
        lambda x: 1 if x[:4] in ['住宿服务', '商务住宅'] else 0
    )
    print(f"住宅类POI数量: {poi_df['is_residential'].sum()}")
    print(f"非住宅类POI数量: {len(poi_df) - poi_df['is_residential'].sum()}")
    
    # 2.2 创建WGS84几何对象
    geometry = [Point(xy) for xy in zip(poi_df['经度_84'], poi_df['纬度_84'])]
    poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry, crs='EPSG:4326')
    
    # 2.3 转换到UTM50N (EPSG:32650)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    poi_gdf['geometry'] = [Point(transformer.transform(pt.x, pt.y)) for pt in poi_gdf.geometry]
    poi_gdf.crs = CRS.from_epsg(32650)
    print(f"转换后POI坐标系验证: {poi_gdf.crs}")

    # 3. 创建与栅格对齐的分析网格
    def create_aligned_grid():
        """生成与输入栅格完全对齐的矢量网格"""
        with rasterio.open(config['building_raster']) as src:
            # 获取栅格行列数
            width = src.width
            height = src.height
            transform = src.transform
            
            # 生成网格单元几何
            grid_geoms = []
            grid_ids = []
            for row in range(height):
                for col in range(width):
                    # 计算每个格子的地理坐标
                    xmin, ymin = transform * (col, row+1)  # 注意y方向
                    xmax, ymax = transform * (col+1, row)
                    grid_geoms.append(box(xmin, ymin, xmax, ymax))
                    grid_ids.append(row * width + col)  # 行优先索引
        
        return gpd.GeoDataFrame({'grid_id': grid_ids, 'geometry': grid_geoms}, crs=src.crs)

    grid_gdf = create_aligned_grid()
    print(f"创建分析网格: {len(grid_gdf)}个单元")

    # 4. 精确空间连接
    def spatial_join_precision():
        """带坐标容差的空间连接"""
        # 设置容差为半个像素大小
        tolerance = raster_transform.a / 2
        
        # 执行空间连接
        joined = gpd.sjoin(poi_gdf, grid_gdf, how='inner', predicate='intersects')
        print(f"成功匹配{len(joined)}个POI点")
        
        # 统计每个网格的住宅和非住宅POI
        grid_poi_stats = pd.DataFrame(index=grid_gdf.index)
        
        # 住宅类POI数量
        residential = joined[joined['is_residential'] == 1].groupby('index_right').size()
        grid_poi_stats['residential_poi'] = residential.reindex(grid_gdf.index).fillna(0)
        
        # 非住宅类POI数量
        non_residential = joined[joined['is_residential'] == 0].groupby('index_right').size()
        grid_poi_stats['non_residential_poi'] = non_residential.reindex(grid_gdf.index).fillna(0)
        
        # 合并到网格
        return grid_gdf.join(grid_poi_stats)

    # 获取带POI信息的网格
    grid_poi = spatial_join_precision()
    print(f"POI统计完成，列包括: {grid_poi.columns.tolist()}")

    # 5. 建筑密度数据处理
    def extract_building_density(input_grid):
        """从栅格中提取密度值到网格
        
        Args:
            input_grid: 输入网格GeoDataFrame，保留其中已有的列
        """
        result = input_grid.copy()  # 创建副本避免修改原始数据
        
        with rasterio.open(config['building_raster']) as src:
            # 将栅格数据读取为数组
            data = src.read(1)
            
            # 展平数组并与网格对应
            result['building_density'] = data.flatten()
            
        print(f"建筑密度提取完成，列包括: {result.columns.tolist()}")
        return result

    # 使用带POI的网格添加建筑密度
    grid_data = extract_building_density(grid_poi)  # 修改这里，传入grid_poi

    # 6. 计算HSI指数
    def calculate_hsi(grid):
        """住房供给潜力指数计算"""
        # 检查必要的列是否存在
        required_cols = ['residential_poi', 'non_residential_poi', 'building_density']
        for col in required_cols:
            if col not in grid.columns:
                raise KeyError(f"缺少计算HSI所需的列: {col}")
        
        # 6.1 处理住宅比例
        total_poi = grid['residential_poi'] + grid['non_residential_poi']
        # 避免除零错误
        grid['res_ratio'] = np.where(
            total_poi > 0,
            grid['residential_poi'] / total_poi,
            0  # 如果没有POI，住宅比例设为0
        )
        
        # 6.2 建筑密度标准化
        density_range = grid['building_density'].max() - grid['building_density'].min()
        if density_range > 0:
            grid['density_norm'] = (grid['building_density'] - grid['building_density'].min()) / density_range
        else:
            grid['density_norm'] = 0
        
        # 6.3 合成HSI
        grid['HSI'] = 0.7 * grid['density_norm'] + 0.3 * grid['res_ratio']
        print(f"HSI计算完成，值范围: {grid['HSI'].min():.4f}-{grid['HSI'].max():.4f}，均值: {grid['HSI'].mean():.4f}")
        return grid

    final_grid = calculate_hsi(grid_data)

    # 7. 保存结果为TIF栅格
    def save_to_tif(grid_df, output_path, reference_meta):
        """将结果保存为TIF栅格格式"""
        # 准备元数据
        meta = reference_meta.copy()
        meta.update({
            'dtype': 'float32',
            'nodata': -9999,
            'count': 1
        })
        
        # 创建HSI栅格数组
        height = meta['height']
        width = meta['width']
        hsi_raster = np.full((height, width), meta['nodata'], dtype='float32')
        
        # 填充HSI值
        for idx, row in grid_df.iterrows():
            grid_id = int(row['grid_id'])
            col = grid_id % width
            row_idx = grid_id // width
            hsi_raster[row_idx, col] = row['HSI']
        
        # 写入栅格文件
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(hsi_raster, 1)
            dst.update_tags(HSI_DESCRIPTION="住房供给潜力指数 (0-1)")
        
        print(f"HSI栅格已保存至: {output_path}")
        return output_path

    save_to_tif(final_grid, config['output_hsi'], raster_meta)
    print(f"处理完成: HSI指数已保存为栅格")

if __name__ == "__main__":
    try:
        coordinate_transformation()
    except Exception as e:
        import traceback
        print(f"处理过程中发生错误: {e}")
        print(traceback.format_exc())