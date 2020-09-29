function recon = reconstruct_face(face_im, sorted_eigfaces, avg_face_vec, k)
    [H,W] = size(face_im);
    face_vec = face_im(:);

    trunc_eigfaces = sorted_eigfaces(1:k,:);
    weights = trunc_eigfaces * (face_vec - avg_face_vec.');

    recon = weights' * trunc_eigfaces;
    
    % Adjust the values of the reconstructed matrix to be within [0-255]
    maxel = max(max(recon));
    minel = min(min(recon));
    recon = (recon - minel) / (maxel-minel);
    recon = reshape(recon, [H,W]) * 255;
end

