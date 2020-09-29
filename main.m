data_folder = "att_faces";
num_samples = 400;
ims_per_id = 10;

% data_folder = "att_faces_simple";
% num_samples = 20;
% ims_per_id = 1;

H = 112;
W = 92;

data_matrix = read_ims_into_matrix(data_folder, num_samples, ims_per_id, H, W);

[avg_face_vec,sorted_eigfaces] = eigenfaces(data_matrix);
% size(sorted_eigenfaces)
% return
% TODO: display according to assignment
% num_eigenfaces = 4;
% 
% for i=1:num_eigenfaces
%     ef = sorted_eigfaces(i,:);
%     maxel = max(max(ef));
%     minel = min(min(ef));
%     ef = (ef - minel) / (maxel - minel);
%     disp_ef = reshape(ef, [H,W]) * 255;
%     imshow(uint8(disp_ef));
%     pause;
% end

% close all
% clear
% return
% 
% eigen_vecs = A * V_sorted;
% size(eigen_vecs)
% Sanity check that the mean face looks like a mean face
% mean_face = reshape(avg_face_vec, [H,W]);
% imshow(uint8(mean_face));
% pause;

sample_face = data_matrix(100,:);
sample_face = reshape(sample_face, [H,W]);
imshow(uint8(sample_face));
pause;

ks = [1, 10, 20, 50, 100, 200, 300, 400];
for k=ks
    fprintf("num eigenfaces=%i\n",k);
    recon = reconstruct_face(sample_face, sorted_eigfaces, avg_face_vec, k);
    imshow(uint8(recon));
    pause;
end
close all;
clear;