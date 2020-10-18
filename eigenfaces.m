function [avg_face,sorted_eigfaces] = eigenfaces(data_matrix)
    avg_face = mean(data_matrix);
    
    face_diffs = data_matrix - avg_face; % This is PSI == A.T
    
    % A is [Psi1 Psi2 ... PsiM], where each Psi is mean-subtracted face-vec
    A = face_diffs.';
    
    % Compute eigenvectors and eigenvalues of A.T * A
    [V,D] = eig(A.'*A);
    
    % Sort them in descending order of eigenvalue
    [~,ind] = sort(diag(D),'descend');
    V_sorted = V(:,ind);
    
    % At this point, V_sorted are eigenvectors of A.T*A
    % But we want eigenvectors of A*A.T
    % If u is an eigenvector from A.T*A, then
    % A*u is an eigenvector from A*A.T
    
    % A: features x M
    % V_sorted: M x M
    % sorted_eigenfaces is M x features
    sorted_eigfaces = (A*V_sorted)';

    % Normalizing the eigenfaces to unit length is important!
    % Otherwise, will face numerical errors (probably overflow?)
    row_norms = sqrt(sum(sorted_eigfaces.^2,2));
    sorted_eigfaces = bsxfun(@rdivide,sorted_eigfaces, row_norms);
end

