%% KSOM - Result Analysis

clc;

% Get max value and its index

[max_mean_value,max_mean_index] = max(mat_acc_mean(:));
[I_row, I_col] = ind2sub(size(mat_acc_mean),max_mean_index);

% Compare training models

mat_acc_comp_ksoms = zeros(3,8);

for j = 1:8
    for i = 1:3
        mat_acc_comp_ksoms(i,j) = mat_acc_mean(i,j) - mat_acc_mean(i+3,j);
    end
end

% Compare Labeling Strategies

mat_acc_comp_lbls = zeros(6,8);

for j = 1:8
    for i = 1:3
        mat_acc_comp_lbls(i,j) = mat_acc_mean(i,j) - ...
                                 max(mat_acc_mean(1:3,j));
    end
    for i = 4:6
        mat_acc_comp_lbls(i,j) = mat_acc_mean(i,j) - ...
                                 max(mat_acc_mean(4:6,j));
    end
end

% Compare Kernels

n_times_best = zeros(1,8);
n_times_3_best = zeros(1,8);
bests_3_kernels_per_line = zeros(6,3);

for i = 1:6
    line = mat_acc_mean(i,:);
    [sorted_line,sorted_indexes] = sort(line,'descend');
    bests_3_kernels_per_line(i,:) = sorted_indexes(1:3);
    n_times_best(sorted_indexes(1)) = n_times_best(sorted_indexes(1)) + 1;
    for j = 1:3
        n_times_3_best(sorted_indexes(j)) = n_times_3_best(sorted_indexes(j)) + 1;
    end
end
    


%% SPARK - Result Analysis



%% SPOK - Result Analysis



%% END