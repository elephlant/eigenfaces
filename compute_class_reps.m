% train: #rows=(samples_per_id * len(id_list)), #cols=H*W
% avg_face_vec: H*W-length vector
% sorted_eigfaces: each row is an eigenface of length H*W
% M: number of eigenfaces to use in representing each class
function cls_reps = compute_class_reps(train,samples_per_id,id_list,avg_face_vec,sorted_eigfaces,M)
    % Take only the top M eigenfaces
    trunc_eigfaces = sorted_eigfaces(1:M,:);
    cls_reps = zeros( [length(id_list), M] );
    
    % Compute face-space representations of all training samples
    weights = trunc_eigfaces * (train.' - avg_face_vec.'); % weights: M x length(train)
    weights = weights.'; % weights: length(train) x M
    
    % The 1:samples_per_id rows of weights are the eigenface
    % representations of images of ID=1
    % The (samples_per_id+1):(samples_per_id*2) rows of weights are the
    % eigenface reps of ID=2 images, and so on...
    
    % Average each identity's face-space vectors and store as
    % representative
    for i=1:length(id_list)
        idx = id_list(i);
        seq = ((i-1)*samples_per_id + 1):(i*samples_per_id);
        cls_reps(idx,:) = mean(weights(seq,:), 1);
    end
end

