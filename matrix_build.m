%% 修正后的地理邻接计算（支持parfor）
function [W, valid_mask] = create_geographic_adjacency(raster_path, distance_type)
    % 读取栅格数据及地理参考信息
    [raster, R] = readgeoraster(raster_path);
    [rows, cols] = size(raster);
    
    % 创建有效像元掩膜（排除NoData）
    valid_mask = ~isnan(raster);
    [valid_rows, valid_cols] = find(valid_mask);
    n_valid = nnz(valid_mask);
    
    % 获取像元地理坐标（中心点）
    [x_center, y_center] = cellCenter(R, valid_rows, valid_cols);
    
    % 计算像元地理尺寸
    cell_size = abs([R.CellExtentInLatitude R.CellExtentInLongitude]);
    
    % 确定邻接阈值
    switch lower(distance_type)
        case 'rook'
            max_distance = max(cell_size)*1.1;
        case 'queen'
            max_distance = norm(cell_size)*1.1;
        otherwise
            error('仅支持rook或queen邻接类型');
    end
    
    % 构建空间索引
    points = [x_center, y_center];
    searcher = KDTreeSearcher(points);
    
    % 预分配临时存储
    rows_cell = cell(n_valid, 1);
    cols_cell = cell(n_valid, 1);
    
    % 并行计算邻接关系（修正变量分类）
    parfor i = 1:n_valid
        [idx, dist] = rangesearch(searcher, points(i,:), max_distance);
        neighbors = idx{1}(dist{1} > 0);
        
        % 存储索引而不是直接操作矩阵
        rows_cell{i} = repmat(i, 1, length(neighbors));
        cols_cell{i} = neighbors;
    end
    
    % 合并稀疏矩阵索引
    rows_idx = [rows_cell{:}];
    cols_idx = [cols_cell{:}];
    values = ones(size(rows_idx));
    
    % 构建稀疏矩阵
    W = sparse(rows_idx, cols_idx, values, n_valid, n_valid);
    
    % 行标准化（添加稳定性处理）
    row_sums = sum(W, 2);
    zero_rows = (row_sums == 0);
    row_sums(zero_rows) = 1; % 避免除零
    W = spdiags(1./row_sums, 0, n_valid, n_valid) * W;
    
    fprintf('邻接矩阵密度: %.2f%%\n', 100*nnz(W)/numel(W));
end


%% 保存地理邻接权重
function save_geographic_adjacency(W, valid_mask, ref_raster, output_path)
    % 重建完整栅格矩阵
    full_matrix = nan(size(valid_mask));
    full_matrix(valid_mask) = sum(W, 2);
    
    % 使用2024a增强版geowrite函数
    geowrite(output_path, full_matrix, ref_raster,...
        'CoordinateSystemType', 'projected',...
        'DataType', 'single');
end

% 示例用法：
[W, mask] = create_geographic_adjacency('poi_hhi_output.tif', 'queen');
save_geographic_adjacency(W, mask, 'poi_hhi_output.tif', 'geographic_queen_weights.tif');
