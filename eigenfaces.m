function [avg_face,sorted_eigfaces] = eigenfaces(data_matrix)
    avg_face = mean(data_matrix);
%     avg_face = sum(data_matrix) / 400;
%     fprintf("%d\n", length(data_matrix));
    
    face_diffs = data_matrix - avg_face; % PSI = A.T

%     size(data_matrix)
%     size(avg_face)
%     size(face_diffs)
%       the sizes are correct, implying that the data_matrix - avg_face is
%       correct

    
    A = face_diffs.';
    
    % Compute eigenvectors and values of A.T * A
    [V,D] = eig(A.'*A);
%     norm(V(:,1))
%     norm(V(:,2))
    % Multiply two random columns of V and see what the result is
%     v1 = V(:,2);
%     v2 = V(:,10);
%     dotp = v1' * v2;
%     dotp
%     yup, the column vectors are orthogonal
    
    [~,ind] = sort(diag(D),'descend');
    V_sorted = V(:,ind);
    
%     norm(V_sorted(:,1))
%     norm(V_sorted(:,10))
%     norm(V_sorted(:,100))

%     D_sorted = D(ind,ind);
%     D_sorted = diag(D_sorted);
%     D_sorted
%     Yes, the eigenvalues are sorted

    % A is [Psi1 Psi2 ... PsiM], where each Psi is column vector
    % sorted_eigenfaces is num_samples x num_feats
    sorted_eigfaces = (A*V_sorted)';

    % Normalizing the eigenfaces to unit length is important!
    % Otherwise, will face numerical errors (probably overflow)
    row_norms = sqrt(sum(sorted_eigfaces.^2,2));
    sorted_eigfaces = bsxfun(@rdivide,sorted_eigfaces, row_norms);
end

