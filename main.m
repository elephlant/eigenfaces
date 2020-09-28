data_folder = "att_faces";

num_samples = 400;
H = 112;
W = 92;

data_matrix = read_ims_into_matrix(data_folder, num_samples, H, W);

mean_row = mean(data_matrix);

face_diffs = data_matrix - mean_row; % PSI = A.T
A = face_diffs.';

% size(A)

% Compute eigenvectors and values of 
[V,D] = eig(A.'*A);
% size(V)
% size(D)
[~,ind] = sort(diag(D),'descend');
D_sorted = D(ind,ind);
V_sorted = V(:,ind);

sorted_eigenfaces = V_sorted.' * A.';

num_eigenfaces = 10;

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