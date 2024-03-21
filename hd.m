% 读取数据集的路径
dataset_folder = 'E:\dataset\MM-WHS-2017\nnnn'; % 更改为您数据集的路径

% 获取所有的子文件夹（每个子文件夹代表一个类别）
subfolders = dir(dataset_folder);
subfolders = subfolders([subfolders.isdir] & ~strcmp({subfolders.name},'.') & ~strcmp({subfolders.name},'..'));

num_classes = length(subfolders);

% 初始化特征矩阵和Y
featureMatrix = [];
Y = [];

for i = 1:num_classes
    class_folder = fullfile(dataset_folder, subfolders(i).name);
    
    % 获取当前类别下的所有JPG图片
    img_files = dir(fullfile(class_folder, '*.png'));
    
    for j = 1:length(img_files)
        % 读取图片
        img_path = fullfile(class_folder, img_files(j).name);
        img = imread(img_path);
        
        % 如果图片是彩色的，则转换为灰度
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % 提取灰度直方图作为特征
        features = hist(double(img(:)), 256);
        features = features / sum(features); % 归一化
        
        % 存储特征和标签
        featureMatrix = [featureMatrix; features];
        Y = [Y; i];
    end
end

% 将特征矩阵包装到一个cell元素中
X = {featureMatrix};

% 保存为.mat格式
save('hd.mat', 'X', 'Y');
