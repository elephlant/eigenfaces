function recon = reconstruct_face(face_im, sorted_eigfaces, avg_face_vec, k)
    [H,W] = size(face_im);
    % Flatten the face image into a column vector
    face_vec = face_im(:);

    % Use only the first k eigenfaces for reconstruction
    trunc_eigfaces = sorted_eigfaces(1:k,:);
    % Compute the weight contribution of each of the k eigenfaces
    weights = trunc_eigfaces * (face_vec - avg_face_vec.');

    % Compute weighted-sum of k eigenfaces to reconstruct the face
    recon = weights' * trunc_eigfaces;
    
    % Adjust the values of the reconstructed matrix to be within [0-255]
    maxel = max(max(recon));
    minel = min(min(recon));
    recon = (recon - minel) / (maxel-minel);
    recon = reshape(recon, [H,W]) * 255;
end

