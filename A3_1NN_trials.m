data_folder = "att_faces";
num_ids = 40;
ims_per_id = 10;
train_samples_per_id = 4;
test_samples_per_id = ims_per_id - train_samples_per_id;
H = 112;
W = 92;

num_trials = 10;

% M is in [1, num_ids * train_samples_per_id]
% In this specific case, [1,160]
Ms=[1,2,3,5,10,25,50,100,160];
m_scores = zeros( size(Ms) );
for i=1:length(Ms)
    M = Ms(i);
    trial_scores = zeros( [num_trials,1] );
    for j=1:num_trials
        % train: (num_ids * train_samples_per_id) rows, cols = H*W
        % test: (num_ids * test_samples_per_id) rows, cols = H*W
        % id_list: vector of num_ids length
        [train,test,id_list] = read_data(data_folder, num_ids, ims_per_id, train_samples_per_id, H, W);

        [avg_face_vec,sorted_eigfaces] = eigenfaces(train);
        cls_reps = compute_class_reps(train,train_samples_per_id,id_list,avg_face_vec,sorted_eigfaces,M);
        [~, acc] = nn_trial(test,test_samples_per_id,id_list,avg_face_vec,sorted_eigfaces,M,cls_reps);
        trial_scores(j) = acc;
    end
    m_scores(i) = mean(trial_scores, 'all');
    fprintf("M=%i, mean accuracy (over 10-trials): %.2f%%\n", M, 100.0 * m_scores(i));
end

plot(Ms, m_scores, '-o', 'LineWidth', 3);
title("Plot of No. Eigenfaces vs. FR-Accuracy");
xlabel("No. Eigenfaces Used");
ylabel("Facial Recognition Accuracy (Over 10 Trials)");
pause;
close all;
clear;