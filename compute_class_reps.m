function cls_reps = compute_class_reps(train,samples_per_id,id_list,avg_face_vec,sorted_eigfaces,M)
    trunc_eigfaces = sorted_eigfaces(1:M,:);
    cls_reps = zeros( [length(id_list), M] );
    
    % Compute face-space representations of all training samples
    weights = trunc_eigfaces * (train.' - avg_face_vec.');
    weights = weights.';
    
    % Average each identity's face-space vectors and store as
    % representative
    for i=1:length(id_list)
        idx = id_list(i);
        seq = ((i-1)*samples_per_id + 1):(i*samples_per_id);
        cls_reps(idx,:) = mean(weights(seq,:), 1);
    end
end

