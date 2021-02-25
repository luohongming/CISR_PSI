

origin_path = '/media/luo/data/data/super-resolution/DIV2K+/origin';
path = '/media/luo/data/data/super-resolution/DIV2K+';

img_names = dir(origin_path);

Q_list = [10, 20, 30, 40, 50, 60, 70, 80, 90];
% Q_list = [0];


sr_factor = [2, 3, 4];


for k = 3: length(img_names)
    img_name = img_names(k).name;
    img = imread(fullfile(img_names(k).folder, img_name));
    

    for i = 1:length(sr_factor)
        img_ = modcrop(img, sr_factor(i));
        sr_path = fullfile(path, strcat('x', num2str(sr_factor(i))));
        img_down = imresize(img_, 1/sr_factor(i), 'bicubic');
        for Q = Q_list
            jpeg_path = fullfile(sr_path, strcat('jpeg', num2str(Q)));
            if ~exist(jpeg_path, 'dir')
                mkdir(jpeg_path);
            end
            
            if Q == 0
                save_name = strcat(img_name(1:end-4), '.png');
                save = fullfile(jpeg_path, save_name)
                imwrite(img_down, save);
            else
                save_name = strcat(img_name(1:end-4), '.jpg');
                save = fullfile(jpeg_path, save_name)
                imwrite(img_down, save, 'jpg', 'quality', Q);
            end

        end
    end

end