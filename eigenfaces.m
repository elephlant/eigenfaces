function [avg_face,sorted_eigenfaces] = eigenfaces(data_matrix)
    avg_face = mean(data_matrix);

    face_diffs = data_matrix - avg_face; % PSI = A.T
    A = face_diffs.';

    % Compute eigenvectors and values of A.T * A
    [V,D] = eig(A.'*A);
    [~,ind] = sort(diag(D),'descend');
%     D_sorted = D(ind,ind);
    V_sorted = V(:,ind);

    sorted_eigenfaces = V_sorted.' * A.';
end

