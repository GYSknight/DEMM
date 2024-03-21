function shape_features = optimized_shape_feature_extraction_function_v2(img)
    % 使用Canny边缘检测
    edges = edge(img, 'Canny');
    
    % 使用bwlabel找到连通区域
    [labeled_img, ~] = bwlabel(edges, 8);
    
    % 使用regionprops计算每个连通区域的面积和周长
    stats = regionprops(labeled_img, 'Area', 'Perimeter');
    areas = [stats.Area];
    perimeters = [stats.Perimeter];
  
    % 计算全局面积和周长
    global_area = sum(areas);
    global_perimeter = sum(perimeters);
    
    % 计算矩形度和圆度
    rectangularity = global_area / (global_perimeter^2);
    roundness = (4 * pi * global_area) / (global_perimeter^2);
        
    % 结果
    shape_features = [global_area, global_perimeter, rectangularity, roundness];
end
