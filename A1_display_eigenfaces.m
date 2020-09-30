data_folder = "att_faces";
num_ids = 40;
ims_per_id = 10;
train_samples_per_id = 10;
H = 112;
W = 92;

% train: (num_ids * train_samples_per_id) rows, cols = H*W
[train,~,~] = read_data(data_folder, num_ids, ims_per_id, train_samples_per_id, H, W);

[~,sorted_eigfaces] = eigenfaces(train);

% Display according to assignment instructions
num_eigenfaces_disp = 10;

composite = zeros( [H,W*10] );

for i=1:num_eigenfaces_disp
    ef = sorted_eigfaces(i,:);
    % Map each vector to [0,1], for display purposes
    maxel = max(max(ef));
    minel = min(min(ef));
    ef = (ef - minel) / (maxel - minel);
    disp_ef = reshape(ef, [H,W]);
    composite(:,(W*(i-1) + 1):(W*i)) = disp_ef;
end

imshow(composite);
title("Part I: First ten eigenfaces (left-to-right).");
pause;
close all;
clear;
