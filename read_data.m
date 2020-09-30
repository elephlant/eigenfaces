function [train, test, id_list] = read_data(data_folder, num_ids, ims_per_id, samples_per_id, H, W)
    % Read all images into a large matrix
    num_feats = H*W;
    num_ims = num_ids * ims_per_id;
    num_samples = num_ids * samples_per_id;
    train = double(zeros( [num_samples, num_feats] ));
    test = double(zeros( [num_ims - num_samples, num_feats] ));
    id_list = zeros( [num_ids,1] );
    
    % Get a list of folders that start with "s" in the root data folder.
    files = dir(fullfile(data_folder, "s*"));
    % Get a logical vector that tells which is a directory.
    dir_flags = [files.isdir];
    % Extract only those that are directories.
    id_folders = files(dir_flags);
    
    train_idx = 0;
    test_idx = 0;
    for k = 1 : length(id_folders)
        id_folder = id_folders(k).name;
        curr_id = str2double(extractAfter(id_folder,"s"));
        id_list(k) = int32(curr_id);
        % Traverse the folder to read each image inside
        full_sub_folder = fullfile(data_folder, id_folder, "*.pgm");
        pgm_files = dir(full_sub_folder);
        % Select a number of images from the person's folder
        choices = randsample(length(pgm_files), samples_per_id);
        for j =1:length(pgm_files)
            pgm_file = pgm_files(j).name;
            pgm_fp = fullfile(data_folder, id_folder, pgm_file);
            im = imread(pgm_fp);
            if ismember(j, choices)
                train_idx = train_idx + 1;
                train(train_idx,:) = im(:);
            else
                test_idx = test_idx + 1;
                test(test_idx,:) = im(:);
            end
        end
    end
end

