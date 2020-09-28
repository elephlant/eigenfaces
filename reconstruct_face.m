function recon = reconstruct_face(face_im, sorted_eigenfaces, average_face_vec, k)
    [H,W] = size(face_im);
    face_vec = face_im(:);
    recon = zeros( size(face_vec) );
    weights = sorted_eigenfaces * (face_vec - average_face_vec).';
    for i=1:k
        recon = recon + (weights(i) * sorted_eigenfaces(i,:));
    end
    recon = reshape(recon, [H,W]);
end

