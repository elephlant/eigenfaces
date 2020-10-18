function [predictions, acc] = nn_trial(test,samples_per_id,id_list,avg_face_vec,sorted_eigfaces,M,cls_reps)
    if size(test,1) <= 0
        fprintf("Empty test set.\n");
        predictions = [];
        acc = 0;
        return;
    end
        
    trunc_eigfaces = sorted_eigfaces(1:M,:);
    
    % Each column is a test sample projected on M-eigenfaces
%     test_reps = trunc_eigfaces * (test.' - avg_face_vec.');
    test_reps = trunc_eigfaces * (test - avg_face_vec).';
    % Each row is a test sample now (for use in pdist2)
    test_reps = test_reps.';
    
    % The jth column contains all class scores for the jth test sample
    dists = pdist2(cls_reps, test_reps, 'euclidean');
    
    % length(test) = length(id_list) * samples_per_id
    labels = repelem(id_list, samples_per_id);
    
    % The min() is the key to 1-NN classification
    [~,predictions] = min(dists);
    
    acc = sum(labels(:) == predictions(:)) / double(size(test,1));
end

