import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point
import os
import sys
# 在脚本最开头添加
print("脚本开始执行...")
print(f"Python版本: {sys.version}")

def calculate_hhi(df, poi_columns):
    """
    计算每个空间单元的HHI指数
    :param df: 包含POI类型数量的DataFrame
    :param poi_columns: POI类型字段列表
    :return: 包含HHI指数的Series
    """
    # 处理POI总数为0的单元格
    total_pois = df[poi_columns].sum(axis=1)
    # 对于总数为0的栅格，设置HHI为0
    if (total_pois == 0).any():
        print(f"警告: 发现{(total_pois == 0).sum()}个POI总数为0的栅格")
    
    # 对于非零总数的栅格计算HHI
    valid_mask = total_pois > 0
    proportions = pd.DataFrame(0, index=df.index, columns=poi_columns)
    proportions.loc[valid_mask] = df.loc[valid_mask, poi_columns].div(
        total_pois[valid_mask], axis=0
    )
    
    # HHI计算 (1 - SUM(proportion^2))
    hhi = 1 - (proportions ** 2).sum(axis=1)
    return hhi

def read_poi_csv(csv_path, x_col='x', y_col='y', type_col='type', crs='EPSG:4326', to_crs=None):
    """
    读取POI CSV文件并转换为GeoDataFrame
    :param csv_path: CSV文件路径
    :param x_col: X坐标列名
    :param y_col: Y坐标列名
    :param type_col: POI类型列名
    :param crs: 坐标参考系统
    :param to_crs: 目标坐标系(如果需要转换)
    :return: POI的GeoDataFrame
    """
    # 读取CSV
    poi_df = pd.read_csv(csv_path)
    print(f"CSV列名: {poi_df.columns.tolist()}")
    print(f"读取到{len(poi_df)}行POI数据")
    
    # 创建几何对象
    geometry = [Point(xy) for xy in zip(poi_df[x_col], poi_df[y_col])]
    
    # 创建GeoDataFrame
    poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry, crs=crs)
    
    # 如果需要转换坐标系
    if to_crs:
        print(f"转换坐标系从{crs}到{to_crs}")
        poi_gdf = poi_gdf.to_crs(to_crs)
    
    return poi_gdf

def read_grid_raster(raster_path, resolution=None):
    """
    读取栅格文件并转换为矢量格子
    :param raster_path: 栅格文件路径
    :param resolution: 如需重采样，提供新的分辨率元组(xres, yres)
    :return: 栅格元数据和矢量格子GeoDataFrame
    """
    with rasterio.open(raster_path) as src:
        # 获取栅格元数据
        meta = src.meta
        
        # 创建栅格格子的矢量表示
        height, width = src.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        
        # 获取每个格子的坐标边界
        xs, ys = rasterio.transform.xy(src.transform, rows.flatten(), cols.flatten())
        xres, _, _, _, _, yres = src.transform.to_gdal()
        
        # 创建格子几何对象
        grid_geoms = []
        grid_ids = []
        
        for i, (x, y) in enumerate(zip(xs, ys)):
            # 创建栅格格子的多边形 (左上、右上、右下、左下)
            minx = x
            maxx = x + abs(xres)
            miny = y + yres  # 注意y方向是负的
            maxy = y
            
            # 创建多边形
            from shapely.geometry import box
            polygon = box(minx, miny, maxx, maxy)
            grid_geoms.append(polygon)
            grid_ids.append(i)
    
    # 创建栅格格子的GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({
        'grid_id': grid_ids,
        'geometry': grid_geoms
    }, crs=src.crs)
    
    return meta, grid_gdf

def poi_to_grid_hhi(poi_gdf, grid_gdf, poi_type_col, poi_categories):
    """
    计算每个栅格内的POI混合度
    :param poi_gdf: POI点的GeoDataFrame
    :param grid_gdf: 栅格格子的GeoDataFrame
    :param poi_type_col: POI类型的字段名
    :param poi_categories: POI类型列表
    :return: 包含HHI的栅格GeoDataFrame
    """
    # 空间连接POI点到格子
    joined = gpd.sjoin(poi_gdf, grid_gdf, how='inner', predicate='within')
    
    # 统计每个格子内各类POI的数量
    poi_counts = pd.DataFrame(index=grid_gdf.index)
    
    for category in poi_categories:
        # 对于每个栅格和类别，计算数量
        cat_counts = joined[joined[poi_type_col] == category].groupby('index_right').size()
        poi_counts[category] = cat_counts.reindex(grid_gdf.index).fillna(0)
    
    # 合并到栅格GeoDataFrame
    result_gdf = grid_gdf.copy()
    for category in poi_categories:
        result_gdf[category] = poi_counts[category]
    
    # 计算每个格子的HHI
    result_gdf['hhi'] = calculate_hhi(result_gdf, poi_categories)
    
    return result_gdf

def hhi_to_raster(grid_hhi_gdf, output_path, reference_meta, value_col='hhi'):
    """
    将HHI结果写入栅格文件
    :param grid_hhi_gdf: 包含HHI的栅格GeoDataFrame
    :param output_path: 输出栅格文件路径
    :param reference_meta: 参考栅格的元数据
    :param value_col: 要写入栅格的值列名
    """
    # 准备栅格元数据
    meta = reference_meta.copy()
    meta.update({
        'dtype': 'float32',
        'nodata': -9999
    })
    
    # 创建栅格化数据
    shapes = [(geom, value) for geom, value in zip(grid_hhi_gdf.geometry, grid_hhi_gdf[value_col])]
    
    # 栅格化
    with rasterio.open(output_path, 'w', **meta) as dst:
        raster = rasterize(
            shapes=shapes,
            out_shape=(meta['height'], meta['width']),
            transform=meta['transform'],
            fill=-9999,
            dtype='float32'
        )
        dst.write(raster, 1)
    
    print(f"HHI栅格已保存至: {output_path}")

def main(poi_csv_path, grid_raster_path, output_raster_path, 
         x_col='x', y_col='y', type_col='type', poi_categories=None):
    """
    主函数，处理POI数据并计算栅格HHI
    """
    # 读取栅格以获取坐标系
    print(f"读取栅格: {grid_raster_path}")
    raster_meta, grid_gdf = read_grid_raster(grid_raster_path)
    
    # 读取POI数据并转换到栅格坐标系
    print(f"读取POI数据: {poi_csv_path}")
    poi_gdf = read_poi_csv(poi_csv_path, x_col=x_col, y_col=y_col, 
                         type_col=type_col, crs='EPSG:4326', 
                         to_crs=grid_gdf.crs)
    
    # 如果未提供POI类别，则从数据中获取
    if poi_categories is None:
        poi_categories = poi_gdf[type_col].unique().tolist()
        print(f"从数据中识别的POI类别: {poi_categories}")
    else:
        # 检查类别是否在数据中存在
        actual_categories = set(poi_gdf[type_col].unique())
        missing = set(poi_categories) - actual_categories
        if missing:
            print(f"警告: 以下POI类别在数据中不存在: {missing}")
            # 只使用实际存在的类别
            poi_categories = [cat for cat in poi_categories if cat in actual_categories]
    
    # 计算HHI
    print("计算栅格HHI...")
    hhi_gdf = poi_to_grid_hhi(poi_gdf, grid_gdf, type_col, poi_categories)
    
    # 输出统计信息
    print(f"栅格总数: {len(hhi_gdf)}")
    print(f"有POI的栅格数: {(hhi_gdf[poi_categories].sum(axis=1) > 0).sum()}")
    print(f"HHI均值: {hhi_gdf['hhi'].mean():.4f}")
    
    # 保存结果为栅格
    print(f"保存HHI栅格: {output_raster_path}")
    hhi_to_raster(hhi_gdf, output_raster_path, raster_meta)
    
    return hhi_gdf

if __name__ == "__main__":
    
    try:
        # 修复路径格式问题
        poi_csv_path = r"D:\houseuse\yga_poi_file_84\深圳市_poi_84.csv"
        grid_raster_path = r"D:\houseuse\poi_stats_poi_counts.tif"
        output_raster_path = r".\poi_hhi_output.tif"
        
        # 先检查文件是否存在
        if not os.path.exists(poi_csv_path):
            print(f"错误: POI文件不存在 {poi_csv_path}")
            exit(1)
        if not os.path.exists(grid_raster_path):
            print(f"错误: 栅格文件不存在 {grid_raster_path}")
            exit(1)
            
        # 读取CSV文件前先查看内容
        df_sample = pd.read_csv(poi_csv_path, nrows=5)
        print("CSV文件前5行:")
        print(df_sample)
        
        # 检查CSV文件的列名
        print(f"CSV列名: {df_sample.columns.tolist()}")
        
        # 假设列名可能是这些
        x_col = 'lng' if 'lng' in df_sample.columns else 'x' if 'x' in df_sample.columns else '经度'
        y_col = 'lat' if 'lat' in df_sample.columns else 'y' if 'y' in df_sample.columns else '纬度'
        type_col = 'type' if 'type' in df_sample.columns else '类型' if '类型' in df_sample.columns else '分类'
        
        print(f"使用坐标列: x={x_col}, y={y_col}, 类型={type_col}")
        
        # 自动检测POI类别
        poi_categories = None  # 设为None让程序自动从数据获取
        
        # 执行处理
        result_gdf = main(poi_csv_path, grid_raster_path, output_raster_path, 
                        x_col=x_col, y_col=y_col, type_col=type_col,
                        poi_categories=poi_categories)
                        
        print("处理完成！")
    except Exception as e:
        import traceback
        print(f"发生错误: {e}")
        print(traceback.format_exc())