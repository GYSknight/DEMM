mypool = gcp;
mypool.IdleTimeout = inf;  % 设置空闲超时为无穷大

% 读取数据集的路径
dataset_folder = 'E:\dataset\MM-WHS-2017\nnnn'; % 更改为您的数据集路径

% 获取所有的子文件夹
subfolders = dir(dataset_folder);
subfolders = subfolders([subfolders.isdir]); % 仅保留文件夹
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % 剔除.和..文件夹

% 初始化特征矩阵和标签
X = cell(1, 1);
Y = [];

% 遍历每个类别并处理其中的图片
total_images = 0; % 用于记录总图片数
for class_label = 1:length(subfolders)
    class_folder = subfolders(class_label).name;
    class_path = fullfile(dataset_folder, class_folder);
    
    % 获取当前类别文件夹下的所有JPG图片
    image_files = dir(fullfile(class_path, '*.png')); % 假设图像格式为jpg
    total_images = total_images + length(image_files); % 增加总图片数
    
    % 初始化当前类别的特征矩阵和标签
    class_feature_matrix = [];
    class_labels = repmat(class_label, length(image_files), 1);
    
    % 遍历当前类别的图片
    for image_label = 1:length(image_files)
        image_file = image_files(image_label).name;
        image_path = fullfile(class_path, image_file);
        
        % 读取并处理图片
        img = imread(image_path);
        if size(img, 3) == 3
            img = rgb2gray(img); % 如果是彩色图片，转换为灰度图片
        end
        
        % 使用MATLAB内置的纹理特征提取函数，这里使用LBP特征
        feature_vector = double(extractLBPFeatures(img)); % 使用LBP特征提取，并转换为double类型
        
        % 将特征添加到当前类别的特征矩阵
        class_feature_matrix = [class_feature_matrix; feature_vector];
    end
    
    % 将当前类别的特征矩阵放入X
    X{1} = [X{1}; class_feature_matrix]; % X是一个1*1的cell数组，内部包含一个矩阵，确保是double类型
    
    % 将当前类别的标签添加到Y
    Y = [Y; class_labels];
end

% 保存处理后的数据
save('lbp.mat', 'X', 'Y');