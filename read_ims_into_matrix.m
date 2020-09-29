function data_matrix = read_ims_into_matrix(data_folder, num_samples, ims_per_id, H, W)
    % Read all images into a large matrix
    num_feats = H*W;
    data_matrix = double(zeros( [num_samples, num_feats] ));
    
    % Get a list of all files and folders in this folder.
    files = dir(data_folder);
    % Get a logical vector that tells which is a directory.
    dir_flags = [files.isdir];
    % Extract only those that are directories.
    sub_folders = files(dir_flags);
    % Print folder names to command window.
    for k = 1 : length(sub_folders)
        sub_folder = sub_folders(k).name;
        if startsWith(sub_folder, "s")
            % Find the number of the folder
            folder_num = str2num(extractAfter(sub_folder, "s"));
            % Traverse the folder to read each image inside
            full_sub_folder = fullfile(data_folder, sub_folder);
            pgm_files = dir(full_sub_folder);
            for j = 1: length(pgm_files)
                pgm_file = pgm_files(j).name;
                if endsWith(pgm_file, ".pgm")
                    % Read image data
                    pgm_fp = fullfile(data_folder, sub_folder, pgm_file);
                    im = imread(pgm_fp);
                    % Find the number of the image, compute the row index it
                    % belongs to
                    toks = split(pgm_file, ".");
                    pgm_num = str2num( toks{1} );
                    mat_idx = (folder_num - 1) * ims_per_id + pgm_num;
%                     fprintf("%d: %s\n", mat_idx, pgm_fp);
                    data_matrix(mat_idx,:) = im(:)';
%                     size(data_matrix(mat_idx,:))
%                     size(im(:)')
    %                 fprintf("subfolder = %s, pgm_file = %s\n", sub_folder, pgm_file);
                end
            end
    %         fprintf('Sub folder #%d = %s\n', k, full_sub_folder);
        end
    end
end

