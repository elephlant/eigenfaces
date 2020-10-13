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
[avg_face_vec,sorted_eigfaces] = eigenfaces(train);

% Number of faces to display (we will pick randomly)
num_faces = 5;
% Different numbers of eigenfaces to use in reconstruction
Ks = [1, 10, 20, 50, 100, 200, 399];

% Randomly choose faces
chosen_faces = randsample(size(train,1), num_faces);

composite = uint8(zeros( [num_faces*H,(length(Ks)+1)*W] ));

for i=1:length(chosen_faces)
    face_idx = chosen_faces(i);
    chosen_face = train(face_idx,:);
    chosen_face = reshape(chosen_face, [H,W]);
    
    % Plot the original face
    composite((H*(i-1))+1:H*i,1:W) = uint8(chosen_face);

    % Plot the reconstruction of the face using the range of k-values
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