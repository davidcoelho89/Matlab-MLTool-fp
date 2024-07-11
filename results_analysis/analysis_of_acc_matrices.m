%% KSOM - Result Analysis

clc;

% %%%%%%%%%%%%%%%%%%%%%%% Get max value and its index

[max_mean_value,max_mean_index] = max(mat_acc_mean(:));
[max_row, max_col] = ind2sub(size(mat_acc_mean),max_mean_index);

best_result_str1 = 'best result =';
if (max_row <= 3)
    best_result_str2 = ' EF-KSOM,';
else
    best_result_str2 = ' GD-KSOM,';
end

if (mod(max_row,3) == 1)
    best_result_str3 = 'MV,';
elseif (mod(max_row,3) == 2)
    best_result_str3 = 'AD,';
else
    best_result_str3 = 'MD,';
end

if(max_col == 1)
    best_result_str4 = 'LIN';
elseif(max_col == 2)
    best_result_str4 = 'GAU';
elseif(max_col == 3)
    best_result_str4 = 'POL';
elseif(max_col == 4)
    best_result_str4 = 'EXP';
elseif(max_col == 5)
    best_result_str4 = 'CAU';
elseif(max_col == 6)
    best_result_str4 = 'LOG';
elseif(max_col == 7)
    best_result_str4 = 'SIG';
elseif(max_col == 8)
    best_result_str4 = 'KMO';
end

best_result_str = strcat(best_result_str1,best_result_str2,...
                         best_result_str3,best_result_str4);
disp(best_result_str);

% %%%%%%%%%%%%%%%%%%%%%%% Compare training models

mat_acc_comp_ksoms = zeros(3,8);

for j = 1:8
    for i = 1:3
        mat_acc_comp_ksoms(i,j) = mat_acc_mean(i,j) - mat_acc_mean(i+3,j);
    end
end

mat_acc_comp_ksoms_sum = sum(sum(mat_acc_comp_ksoms));
mat_acc_comp_ksoms_nEF = sum(sum(mat_acc_comp_ksoms > 0));
mat_acc_comp_ksoms_nGD = sum(sum(mat_acc_comp_ksoms < 0));

if (mat_acc_comp_ksoms_nEF > mat_acc_comp_ksoms_nGD)
    disp('best model = KSOM-EF');
else
    disp('best model = KSOM-GD');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare Labeling Strategies

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

mat_acc_comp_lbls_best = (mat_acc_comp_lbls == 0);

mat_acc_comp_lbls_MV = sum(mat_acc_comp_lbls_best(1,:)) + ...
                       sum(mat_acc_comp_lbls_best(4,:));
mat_acc_comp_lbls_AD = sum(mat_acc_comp_lbls_best(2,:)) + ...
                       sum(mat_acc_comp_lbls_best(5,:));
mat_acc_comp_lbls_MD = sum(mat_acc_comp_lbls_best(3,:)) + ...
                       sum(mat_acc_comp_lbls_best(6,:));

if(mat_acc_comp_lbls_MV > mat_acc_comp_lbls_AD && ...
   mat_acc_comp_lbls_MV > mat_acc_comp_lbls_MD)
    disp('best lbl = MV');
elseif(mat_acc_comp_lbls_AD > mat_acc_comp_lbls_MV && ...
   mat_acc_comp_lbls_AD > mat_acc_comp_lbls_MD)
    disp('best lbl = AD');
else
    disp('best lbl = MD');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare Kernels

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

[max_n_times_best,best_kernel_1] = max(n_times_best);
[max_n_times_3_best,best_kernel_2] = max(n_times_3_best);

if(best_kernel_1 == 1)
    disp('best kernel = linear');
elseif(best_kernel_1 == 2)
    disp('best kernel = gaussian');
elseif(best_kernel_1 == 3)
    disp('best kernel = poynomial');
elseif(best_kernel_1 == 4)
    disp('best kernel = exponential');
elseif(best_kernel_1 == 5)
    disp('best kernel = cauchy');
elseif(best_kernel_1 == 6)
    disp('best kernel = log');
elseif(best_kernel_1 == 7)
    disp('best kernel = sigmoid');
elseif(best_kernel_1 == 8)
    disp('best kernel = kmod');
end

%% SPARK - Result Analysis



%% SPOK - Result Analysis



%% END