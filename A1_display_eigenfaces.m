% Points to the dataset folder
data_folder = "att_faces";
% Number of unique identities (folders) to expect in the dataset
num_ids = 40;
% Number of images to expect per unique identity
ims_per_id = 10;
% How many do we want to sample from each identity for training? The rest
% go to testing.
train_samples_per_id = 10;
% Image dimensions, so that we know how many features/columns in the design
% matrix - each row is an image sample
H = 112;
W = 92;

% train: #rows=(num_ids * train_samples_per_id), #cols=H*W
[train,~,~] = read_data(data_folder, num_ids, ims_per_id, train_samples_per_id, H, W);

% Derive eigenfaces from the training set. In this case, we use all images
% of the dataset because we're not testing yet.
[~,sorted_eigfaces] = eigenfaces(train);

% Display according to assignment instructions
num_eigenfaces_disp = 10;

composite = zeros( [H,W*10] );

for i=1:num_eigenfaces_disp
    ef = sorted_eigfaces(i,:);
    % Map each vector to [0,1], to display the eigenfaces as double images
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
