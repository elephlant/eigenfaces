data_folder = "att_faces";

num_samples = 400;
H = 112;
W = 92;

data_matrix = read_ims_into_matrix(data_folder, num_samples, H, W);

[avg_face,sorted_eigenfaces] = eigenfaces(data_matrix);

num_eigenfaces = 5;

for i=1:num_eigenfaces
    disp_ef = reshape(sorted_eigenfaces(i,:), [H,W]);
    imshow(uint8(disp_ef));
    pause;
end
% eigen_vecs = A * V_sorted;
% size(eigen_vecs)
% Sanity check that the mean face looks like a mean face
% mean_face = reshape(mean_row, [H,W]);
% imshow(uint8(mean_face));

close all;
clear;