function [avg_face,sorted_eigfaces] = eigenfaces(data_matrix)
    avg_face = mean(data_matrix);
    
    face_diffs = data_matrix - avg_face; % PSI = A.T
    
    A = face_diffs.';
    
    % Compute eigenvectors and eigenvalues of A.T * A
    [V,D] = eig(A.'*A);
    
    % Sort them in descending order of eigenvalue
    [~,ind] = sort(diag(D),'descend');
    V_sorted = V(:,ind);
    
    % A is [Psi1 Psi2 ... PsiM], where each Psi is column vector
    % sorted_eigenfaces is num_samples x num_feats
    sorted_eigfaces = (A*V_sorted)';

    % Normalizing the eigenfaces to unit length is important!
    % Otherwise, will face numerical errors (probably overflow?)
    row_norms = sqrt(sum(sorted_eigfaces.^2,2));
    sorted_eigfaces = bsxfun(@rdivide,sorted_eigfaces, row_norms);
end

