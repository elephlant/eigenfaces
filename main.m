data_folder = "att_faces";
num_ids = 40;
ims_per_id = 10;
train_samples_per_id = 4;
test_samples_per_id = ims_per_id - train_samples_per_id;
H = 112;
W = 92;

% train: (num_ids * train_samples_per_id) rows, cols = H*W
% test: (num_ids * test_samples_per_id) rows, cols = H*W
% id_list: vector of num_ids length
[train,test,id_list] = read_data(data_folder, num_ids, ims_per_id, train_samples_per_id, H, W);

[avg_face_vec,sorted_eigfaces] = eigenfaces(train);

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
% 
% close all
% clear
% return

% M is in [1, num_ids * train_samples_per_id]
% In this specific case, [1,160]
M=100;
cls_reps = compute_class_reps(train,train_samples_per_id,id_list,avg_face_vec,sorted_eigfaces,M);

[~, acc] = nn_trial(test,test_samples_per_id,id_list,avg_face_vec,sorted_eigfaces,M,cls_reps);


% Test: visualize the class reps
% cls_recons = cls_reps * sorted_eigfaces(1:M,:);
% for i=1:4
%     cls_recon = cls_recons(i,:);
%     maxel = max(max(cls_recon));
%     minel = min(min(cls_recon));
%     cls_recon = (cls_recon - minel) / (maxel-minel);
%     cls_recon = reshape( cls_recon, [H,W] );
%     imshow(cls_recon);
%     pause;
% end
% close all;
% clear;
% return
% The results make sense - each class rep does look like a blurred version
% of all it's images


sample_face = train(200,:);
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