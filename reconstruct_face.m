function recon = reconstruct_face(face_im, sorted_eigfaces, avg_face_vec, k)
    [H,W] = size(face_im);
    % Flatten the face image into a column vector
    face_vec = face_im(:);

    % Use only the first k eigenfaces for reconstruction
    trunc_eigfaces = sorted_eigfaces(1:k,:);
    % Project the input face (average adjusted) onto the k orthogonal
    % eigenfaces
    weights = trunc_eigfaces * (face_vec - avg_face_vec.');
    
    % weights: (k,1) column vector
    %          point in k-dimensional space, 
    %          with coordinate axes == orthogonal eigenfaces.

    % Compute weighted-sum of k eigenfaces => face reconstruction
    recon = weights' * trunc_eigfaces;
    
    % recon: (1, H*W) row-vector
    
    % Add back the average face vector
    recon = recon + avg_face_vec;
    
    % Reshape the reconstruction back to 2-D
    recon = reshape(recon, [H,W]);
end

