import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, box
import os
import sys

print("脚本开始执行...")

def calculate_hhi(df, poi_columns, nodata=-9999):
    """
    计算每个空间单元的HHI指数，无POI的单元设置为nodata
    """
    # 处理POI总数为0的单元格
    total_pois = df[poi_columns].sum(axis=1)
    if (total_pois == 0).any():
        print(f"警告: 发现{(total_pois == 0).sum()}个POI总数为0的栅格")
    
    # 对于非零总数的栅格计算HHI
    valid_mask = total_pois > 0
    proportions = pd.DataFrame(0, index=df.index, columns=poi_columns)
    proportions.loc[valid_mask] = df.loc[valid_mask, poi_columns].div(
        total_pois[valid_mask], axis=0
    )
    
    # HHI计算 (1 - SUM(proportion^2))
    hhi = pd.Series(nodata, index=df.index)  # 默认所有值为nodata
    hhi.loc[valid_mask] = 1 - (proportions.loc[valid_mask] ** 2).sum(axis=1)  # 只计算有POI的栅格
    
    return hhi

def read_poi_csv(csv_path, x_col='经度_84', y_col='纬度_84', type_col='类型'):
    """
    读取POI CSV文件并转换为GeoDataFrame
    """
    # 读取CSV
    poi_df = pd.read_csv(csv_path, encoding='gbk')
    print(f"CSV列名: {poi_df.columns.tolist()}")
    print(f"读取到{len(poi_df)}行POI数据")
    
    # 截取类型列第一个分号前的文字作为分类
    poi_df['类型_简化'] = poi_df[type_col].str.split(';').str[0]
    print(f"POI类型统计: {poi_df['类型_简化'].value_counts().head(10)}")
    
    # 创建几何对象
    geometry = [Point(xy) for xy in zip(poi_df[x_col], poi_df[y_col])]
    
    # 创建GeoDataFrame - 明确指定为WGS 84坐标系
    poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry, crs='EPSG:4326')
    print(f"POI坐标系: {poi_gdf.crs}")
    
    return poi_gdf

def read_grid_raster(raster_path):
    """
    修复版栅格读取函数，精确生成栅格格子几何
    """
    with rasterio.open(raster_path) as src:
        # 获取栅格元数据
        meta = src.meta.copy()
        transform = src.transform
        
        # 处理坐标系
        if src.crs is None:
            print("强制设置坐标系为UTM 32650")
            src_crs = rasterio.crs.CRS.from_epsg(32650)
            meta.update(crs=src_crs)
        else:
            src_crs = src.crs
            print(f"原始坐标系: {src_crs}")

        # 获取栅格参数
        x_res = transform.a  # x方向分辨率
        y_res = transform.e  # y方向分辨率（通常为负值）
        print(f"栅格分辨率: X={x_res}, Y={y_res}")

        # 生成行列索引矩阵（从0开始）
        rows, cols = np.indices(src.shape)
        rows = rows.flatten()
        cols = cols.flatten()

        # 计算每个格子的四个顶点坐标
        xs, ys = transform * (cols, rows)            # 左上角
        xe, ye = transform * (cols + 1, rows + 1)    # 右下角

        # 生成几何多边形（修复方向问题）
        grid_geoms = [
            box(left, bottom, right, top)
            for left, top, right, bottom in zip(
                xs, ys, xe + abs(x_res), ye + y_res  # 处理负向分辨率
            )
        ]

        # 创建连续grid_id（行优先顺序）
        grid_ids = np.arange(len(grid_geoms))

        # 验证首尾坐标
        print(f"首格子坐标范围: {grid_geoms[0].bounds}")
        print(f"末格子坐标范围: {grid_geoms[-1].bounds}")
        print(f"理论右下角坐标应接近: {transform * (src.width, src.height)}")

    # 创建GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(
        {'grid_id': grid_ids, 'geometry': grid_geoms},
        crs=src_crs
    )

    # 添加元数据校验
    print(f"生成栅格数: {len(grid_gdf)} (理论应 {src.width * src.height})")
    print(f"坐标系验证: {grid_gdf.crs == src_crs}")
    
    return meta, grid_gdf


def poi_to_grid_hhi(poi_gdf, grid_gdf, poi_categories):
    """
    计算每个栅格内的POI混合度
    """
    # 输出转换前的示例坐标
    print(f"转换前POI点示例: {poi_gdf.iloc[0].geometry}")
    print(f"转换前POI坐标系: {poi_gdf.crs}")
    print(f"栅格格子坐标系: {grid_gdf.crs}")
    
    # 确保POI和栅格使用相同的坐标系
    if poi_gdf.crs != grid_gdf.crs:
        print(f"转换POI坐标系从{poi_gdf.crs}到{grid_gdf.crs}")
        poi_gdf = poi_gdf.to_crs(grid_gdf.crs)
        print(f"转换后POI点示例: {poi_gdf.iloc[0].geometry}")
    
    # 添加缓冲区以处理精度问题
    buffer_distance = 0  # 如果需要，可以设置一个小的缓冲距离
    
    # 空间连接POI点到格子
    print("执行空间连接...")
    joined = gpd.sjoin(poi_gdf, grid_gdf, how='inner', predicate='within')
    print(f"空间连接后的POI点数量: {len(joined)}")
    
    # 如果连接的POI点数量明显少于原始POI点，可能有坐标系或精度问题
    if len(joined) < len(poi_gdf) * 0.9:
        print(f"警告: 大量POI点({len(poi_gdf) - len(joined)})未被分配到栅格格子")
        
        # 尝试使用更宽松的空间关系
        print("尝试使用intersects空间关系...")
        joined = gpd.sjoin(poi_gdf, grid_gdf, how='inner', predicate='intersects')
        print(f"使用intersects后的POI点数量: {len(joined)}")
    
    # 统计每个格子内各类POI的数量
    poi_counts = pd.DataFrame(index=grid_gdf.index)
    
    for category in poi_categories:
        # 对于每个栅格和类别，计算数量
        cat_counts = joined[joined['类型_简化'] == category].groupby('index_right').size()
        poi_counts[category] = cat_counts.reindex(grid_gdf.index).fillna(0)
    
    # 合并到栅格GeoDataFrame
    result_gdf = grid_gdf.copy()
    for category in poi_categories:
        result_gdf[category] = poi_counts[category]
    
    # 计算每个格子的HHI
    result_gdf['hhi'] = calculate_hhi(result_gdf, poi_categories)
    
    return result_gdf

def hhi_to_raster(grid_hhi_gdf, output_path, reference_meta, value_col='hhi', nodata=-9999):
    """
    将HHI结果写入栅格文件，严格参照参考栅格的格式
    """
    # 准备栅格元数据 - 完整复制参考元数据
    meta = reference_meta.copy()
    meta.update({
        'dtype': 'float32',
        'nodata': nodata,  # 明确设置nodata值
        'count': 1
    })
    
    # 获取栅格尺寸
    height = meta['height']
    width = meta['width']
    print(f"创建大小为 {width}x{height} 的输出栅格")
    
    # 创建用于存储结果的空栅格
    raster_data = np.full((1, height, width), nodata, dtype='float32')
    
    # 获取栅格变换
    transform = meta['transform']
    print(f"栅格变换: {transform}")
    print(f"栅格原点: ({transform[2]}, {transform[5]})")
    
    # 确保grid_id存在且正确
    if 'grid_id' not in grid_hhi_gdf.columns:
        print("警告: grid_id不存在，使用几何对象位置进行栅格化")
        # 将所有nodata值的记录排除，避免覆盖默认nodata值
        valid_data = grid_hhi_gdf[grid_hhi_gdf[value_col] != nodata].copy()
        if len(valid_data) < len(grid_hhi_gdf):
            print(f"注意: {len(grid_hhi_gdf) - len(valid_data)}个栅格因无数据被排除")
        
        raster = rasterize(
            shapes=[(geom, value) for geom, value in zip(valid_data.geometry, valid_data[value_col])],
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype='float32',
            all_touched=False
        )
    else:
        # 使用grid_id映射到栅格位置
        print("使用grid_id定位栅格位置")
        raster = np.full((height, width), nodata, dtype='float32')
        
        # 遍历每个格子，将HHI值写入对应位置
        for idx, row in grid_hhi_gdf.iterrows():
            grid_id = row['grid_id']
            if grid_id < width * height and row[value_col] != nodata:
                # 将一维索引转换为二维行列
                col = grid_id % width
                row_idx = grid_id // width
                # 只写入非nodata值
                raster[row_idx, col] = row[value_col]
    
    # 写入栅格
    with rasterio.open(output_path, 'w', **meta) as dst:
        # 确保写入一个波段
        if len(raster.shape) == 2:
            dst.write(raster, 1)
        else:
            dst.write(raster)
        
        # 确保坐标系正确
        if 'crs' in meta and meta['crs'] is not None:
            dst.crs = meta['crs']
        
        # 添加元数据描述
        dst.update_tags(HHI_DESCRIPTION="POI混合度指数 (0-1), 无数据=-9999")
    
    # 输出统计信息
    valid_count = np.sum(raster != nodata)
    print(f"栅格统计: 总单元数={width*height}, 有效值单元数={valid_count}, 无数据单元数={width*height-valid_count}")
    
    return output_path
    
def main():
    """
    主函数，处理POI数据并计算栅格HHI
    """
    try:
        NODATA_VALUE = -9999
        # 设置文件路径
        poi_csv_path = r"D:\houseuse\yga_poi_file_84\深圳市_poi_84.csv"
        grid_raster_path = r"D:\houseuse\poi_stats_poi_counts.tif"
        output_raster_path = r"D:\houseuse\poi_hhi_output.tif"
        
        # 检查文件是否存在
        for path, desc in [(poi_csv_path, "POI CSV"), (grid_raster_path, "栅格")]:
            if not os.path.exists(path):
                print(f"错误: {desc}文件不存在: {path}")
                return
        
        # 读取栅格
        print(f"读取栅格: {grid_raster_path}")
        raster_meta, grid_gdf = read_grid_raster(grid_raster_path)
        
        # 读取POI数据
        print(f"读取POI数据: {poi_csv_path}")
        poi_gdf = read_poi_csv(poi_csv_path)
        
        # 获取前10个最常见的POI类型
        top_poi_types = poi_gdf['类型_简化'].value_counts().head(10).index.tolist()
        print(f"使用前10个最常见的POI类型: {top_poi_types}")
        
        # 计算HHI
        print("计算栅格HHI...")
        hhi_gdf = poi_to_grid_hhi(poi_gdf, grid_gdf, top_poi_types)
        
        # 输出统计信息
        print(f"栅格总数: {len(hhi_gdf)}")
        print(f"有POI的栅格数: {(hhi_gdf[top_poi_types].sum(axis=1) > 0).sum()}")
        print(f"HHI均值: {hhi_gdf['hhi'].mean():.4f}")
        
        # 保存结果为栅格
        output_path = hhi_to_raster(hhi_gdf, output_raster_path, raster_meta)
        print(f"处理完成! 结果保存在: {output_path}")
        
    except Exception as e:
        import traceback
        print(f"处理过程中发生错误: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()