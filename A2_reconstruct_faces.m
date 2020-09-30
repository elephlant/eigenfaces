data_folder = "att_faces";
num_ids = 40;
ims_per_id = 10;
train_samples_per_id = 10;
H = 112;
W = 92;

% train: (num_ids * train_samples_per_id) rows, cols = H*W
[train,~,~] = read_data(data_folder, num_ids, ims_per_id, train_samples_per_id, H, W);

[avg_face_vec,sorted_eigfaces] = eigenfaces(train);

num_faces = 5;
Ks = [1, 10, 20, 50, 100, 200, 399];

% Randomly choose faces
chosen_faces = randsample(size(train,1), num_faces);

composite = uint8(zeros( [num_faces*H,(length(Ks)+1)*W] ));

for i=1:length(chosen_faces)
    face_idx = chosen_faces(i);
    chosen_face = train(face_idx,:);
    chosen_face = reshape(chosen_face, [H,W]);
    
    composite((H*(i-1))+1:H*i,1:W) = uint8(chosen_face);

    for j=1:length(Ks)
        k = Ks(j);
        recon = reconstruct_face(chosen_face, sorted_eigfaces, avg_face_vec, k);
        composite((H*(i-1))+1:H*i,(j*W+1):W*(j+1)) = uint8(recon);
    end
end

imshow(composite);
title("Part II: Original | k=1 | k=10 | k=20 | k=50 | k=100 | k=200 | k=399");
pause;
close all;
clear;