%% KSOM - Result Analysis

clc;

% %%%%%%%%%%%%%%%%%%%%%%% Max and Min Values

disp('max value: ')
disp(max(max(mat_acc_mean)));
disp('min value: ')
disp(min(min(mat_acc_mean)));

% %%%%%%%%%%%%%%%%%%%%%%% Get best combination

[max_mean_value,max_mean_index] = max(mat_acc_mean(:));
[max_row, max_col] = ind2sub(size(mat_acc_mean),max_mean_index);

disp(max_mean_value);

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

cont_times_best = zeros(1,8);
cont_times_3_best = zeros(1,8);
bests_3_kernels_per_line = zeros(6,3);

for i = 1:6
    line = mat_acc_mean(i,:);
    [sorted_line,sorted_indexes] = sort(line,'descend');
    bests_3_kernels_per_line(i,:) = sorted_indexes(1:3);
    cont_times_best(sorted_indexes(1)) = cont_times_best(sorted_indexes(1)) + 1;
    for j = 1:3
        cont_times_3_best(sorted_indexes(j)) = cont_times_3_best(sorted_indexes(j)) + 1;
    end
end

[max_n_times_best,best_kernel_1] = max(cont_times_best);
[max_n_times_3_best,best_kernel_2] = max(cont_times_3_best);

if(best_kernel_2 == 1)
    disp('best kernel = LIN');
elseif(best_kernel_2 == 2)
    disp('best kernel = GAU');
elseif(best_kernel_2 == 3)
    disp('best kernel = POL');
elseif(best_kernel_2 == 4)
    disp('best kernel = EXP');
elseif(best_kernel_2 == 5)
    disp('best kernel = CAU');
elseif(best_kernel_2 == 6)
    disp('best kernel = LOG');
elseif(best_kernel_2 == 7)
    disp('best kernel = SIG');
elseif(best_kernel_2 == 8)
    disp('best kernel = KMOD');
end

%% SPARK - Result Analysis

clc;

% %%%%%%%%%%%%%%%%%%%%%%% Max and Min Values

disp('max value: ')
disp(max(max(mat_acc_mean)));
disp('min value: ')
disp(min(min(mat_acc_mean)));

% %%%%%%%%%%%%%%%%%%%%%%% Get best combination

[max_mean_value,max_mean_index] = max(mat_acc_mean(:));
[max_row, max_col] = ind2sub(size(mat_acc_mean),max_mean_index);

disp(mat_acc_mean == max_mean_value);
disp(sum(sum(mat_acc_mean == max_mean_value)))
disp(max_mean_value);

best_result_str1 = 'best result =';

if(max_row <= 4)
    best_result_str2 = ' ALD,';
elseif(max_row <= 8)
    best_result_str2 = ' COH,';
elseif(max_row <= 12)
    best_result_str2 = ' NOV,';
elseif(max_row <= 16)
    best_result_str2 = ' SUR,';
end

if(mod(max_row,4) == 1 || mod(max_row,4) == 2)
    best_result_str3 = ' DM1,';
else
    best_result_str3 = ' DM2,';
end

if(mod(max_row,2) == 0)
    best_result_str4 = ' KNN,';
else
    best_result_str4 = ' 1NN,';
end

if(max_col == 1)
    best_result_str5 = ' LIN';
elseif(max_col == 2)
    best_result_str5 = ' GAU';
elseif(max_col == 3)
    best_result_str5 = ' POL';
elseif(max_col == 4)
    best_result_str5 = ' EXP';
elseif(max_col == 5)
    best_result_str5 = ' CAU';
elseif(max_col == 6)
    best_result_str5 = ' LOG';
elseif(max_col == 7)
    best_result_str5 = ' SIG';
elseif(max_col == 8)
    best_result_str5 = ' KMO';
end

best_result_str = strcat(best_result_str1,best_result_str2,...
                         best_result_str3,best_result_str4,...
                         best_result_str5);
disp(best_result_str);

% %%%%%%%%%%%%%%%%%%%%%%% Compare Sparsification Methods

mat_acc_comp_sparse = zeros(16,8);
cont_best_sparse = zeros(1,4);
for j = 1:8
    for i = 1:4
        max_value = max([mat_acc_mean(i,j),...
                         mat_acc_mean(i+4,j),...
                         mat_acc_mean(i+8,j),...
                         mat_acc_mean(i+12,j)]);
        mat_acc_comp_sparse(i,j) = mat_acc_mean(i,j) - max_value;
        mat_acc_comp_sparse(i+4,j) = mat_acc_mean(i+4,j) - max_value;
        mat_acc_comp_sparse(i+8,j) = mat_acc_mean(i+8,j) - max_value;
        mat_acc_comp_sparse(i+12,j) = mat_acc_mean(i+12,j) - max_value;
    end
end

cont_best_sparse(1) = sum(sum( mat_acc_comp_sparse(1:4,:) == 0 ));
cont_best_sparse(2) = sum(sum( mat_acc_comp_sparse(5:8,:) == 0 ));
cont_best_sparse(3) = sum(sum( mat_acc_comp_sparse(9:12,:) == 0 ));
cont_best_sparse(4) = sum(sum( mat_acc_comp_sparse(13:16,:) == 0 ));

if (cont_best_sparse(1) == max(cont_best_sparse))
    disp('best sparsification = ALD');
elseif (cont_best_sparse(2) == max(cont_best_sparse))
    disp('best sparsification = COH');
elseif (cont_best_sparse(3) == max(cont_best_sparse))
    disp('best sparsification = NOV');
elseif (cont_best_sparse(4) == max(cont_best_sparse))
    disp('best sparsification = SURP');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare Design Methods

mat_acc_comp_dms = zeros(16,8);

for j = 1:8
    for i = 1:4
        max_value1 = max([mat_acc_mean(4*i-3,j),mat_acc_mean(4*i-1,j)]);
        max_value2 = max([mat_acc_mean(4*i-2,j),mat_acc_mean(4*i,j)]);
        mat_acc_comp_dms(4*i-3,j) = mat_acc_mean(4*i-3,j) - max_value1;
        mat_acc_comp_dms(4*i-1,j) = mat_acc_mean(4*i-1,j) - max_value1;
        mat_acc_comp_dms(4*i-2,j) = mat_acc_mean(4*i-2,j) - max_value2;
        mat_acc_comp_dms(4*i,j) = mat_acc_mean(4*i,j) - max_value2;
    end
end

cont_best_dms = zeros(1,2);
cont_best_dms(1) = sum(sum(mat_acc_comp_dms([1,2,5,6,9,10,13,14],:) == 0));
cont_best_dms(2) = sum(sum(mat_acc_comp_dms([3,4,7,8,11,12,15,16],:) == 0));

if (cont_best_dms(1) >= cont_best_dms(2))
    disp('best DM = 1');
else
    disp('best DM = 2');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare NN x KNN

mat_comp_nns = zeros(16,8);
for j = 1:8
    for i = 1:8
        max_value = max([mat_acc_mean(2*i-1,j),mat_acc_mean(2*i,j)]);
        mat_comp_nns(2*i-1,j) = mat_acc_mean(2*i-1,j) - max_value;
        mat_comp_nns(2*i,j) = mat_acc_mean(2*i,j) - max_value;
    end
end

cont_best_nns = zeros(1,2);
cont_best_nns(1) = sum(sum(mat_comp_nns(1:2:15,:) == 0));
cont_best_nns(2) = sum(sum(mat_comp_nns(2:2:16,:) == 0));

if (cont_best_nns(1) >= cont_best_nns(2))
    disp('best NN = 1-NN');
else
    disp('best NN = K-NN');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare Kernels

cont_times_best = zeros(1,8);
cont_times_3_best = zeros(1,8);
bests_3_kernels_per_line = zeros(16,3);

for i = 1:16
    line = mat_acc_mean(i,:);
    [sorted_line,sorted_indexes] = sort(line,'descend');
    bests_3_kernels_per_line(i,:) = sorted_indexes(1:3);
    cont_times_best(sorted_indexes(1)) = cont_times_best(sorted_indexes(1)) + 1;
    for j = 1:3
        cont_times_3_best(sorted_indexes(j)) = cont_times_3_best(sorted_indexes(j)) + 1;
    end
end

[max_n_times_best,best_kernel_1] = max(cont_times_best);
[max_n_times_3_best,best_kernel_2] = max(cont_times_3_best);

if(best_kernel_2 == 1)
    disp('best kernel = LIN');
elseif(best_kernel_2 == 2)
    disp('best kernel = GAU');
elseif(best_kernel_2 == 3)
    disp('best kernel = POL');
elseif(best_kernel_2 == 4)
    disp('best kernel = EXP');
elseif(best_kernel_2 == 5)
    disp('best kernel = CAU');
elseif(best_kernel_2 == 6)
    disp('best kernel = LOG');
elseif(best_kernel_2 == 7)
    disp('best kernel = SIG');
elseif(best_kernel_2 == 8)
    disp('best kernel = KMOD');
end

%% SPOK - Result Analysis (1) - NN and KNN separated

% %%%%%%%%%%%%%%%%%%%%%%% Max and Min Values

disp('max acc value: ')
disp(max(max(mat_acc_final)));
disp('min acc value: ')
disp(min(min(mat_acc_final)));

mat_err_final = 1 - mat_acc_final;

disp('max error value: ')
disp(max(max(mat_err_final)));
disp('min error value: ')
disp(min(min(mat_err_final)));

% %%%%%%%%%%%%%%%%%%%%%%% Get best combination

[max_final_value,max_final_index] = max(mat_acc_final(:));
[max_row, max_col] = ind2sub(size(mat_acc_final),max_final_index);

disp(mat_acc_final == max_final_value);
disp(sum(sum(mat_acc_final == max_final_value)))
disp(max_final_value);

best_result_str1 = 'best result =';

if(max_row <= 2)
    best_result_str2 = ' ALD,';
elseif(max_row <= 4)
    best_result_str2 = ' COH,';
elseif(max_row <= 6)
    best_result_str2 = ' NOV,';
elseif(max_row <= 8)
    best_result_str2 = ' SUR,';
end

if(mod(max_row,2) == 0)
    best_result_str3 = ' KNN,';
else
    best_result_str3 = ' 1NN,';
end

if(max_col == 1)
    best_result_str4 = ' LIN';
elseif(max_col == 2)
    best_result_str4 = ' GAU';
elseif(max_col == 3)
    best_result_str4 = ' POL';
elseif(max_col == 4)
    best_result_str4 = ' EXP';
elseif(max_col == 5)
    best_result_str4 = ' CAU';
elseif(max_col == 6)
    best_result_str4 = ' LOG';
elseif(max_col == 7)
    best_result_str4 = ' SIG';
elseif(max_col == 8)
    best_result_str4 = ' KMO';
end

best_result_str = strcat(best_result_str1,best_result_str2,...
                         best_result_str3,best_result_str4);
disp(best_result_str);

% %%%%%%%%%%%%%%%%%%%%%%% Compare Sparsification Methods

mat_acc_comp_sparse = zeros(8,8);
cont_best_sparse = zeros(1,4);

for j = 1:8
    for i = 1:2
        max_value = max([mat_acc_final(i,j),...
                         mat_acc_final(i+2,j),...
                         mat_acc_final(i+4,j),...
                         mat_acc_final(i+6,j)]);
        mat_acc_comp_sparse(i,j) = mat_acc_final(i,j) - max_value;
        mat_acc_comp_sparse(i+2,j) = mat_acc_final(i+2,j) - max_value;
        mat_acc_comp_sparse(i+4,j) = mat_acc_final(i+4,j) - max_value;
        mat_acc_comp_sparse(i+6,j) = mat_acc_final(i+6,j) - max_value;
    end
end

cont_best_sparse(1) = sum(sum( mat_acc_comp_sparse(1:2,:) == 0 ));
cont_best_sparse(2) = sum(sum( mat_acc_comp_sparse(3:4,:) == 0 ));
cont_best_sparse(3) = sum(sum( mat_acc_comp_sparse(5:6,:) == 0 ));
cont_best_sparse(4) = sum(sum( mat_acc_comp_sparse(7:8,:) == 0 ));

if (cont_best_sparse(1) == max(cont_best_sparse))
    disp('best sparsification = ALD');
elseif (cont_best_sparse(2) == max(cont_best_sparse))
    disp('best sparsification = COH');
elseif (cont_best_sparse(3) == max(cont_best_sparse))
    disp('best sparsification = NOV');
elseif (cont_best_sparse(4) == max(cont_best_sparse))
    disp('best sparsification = SURP');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare NN x KNN

mat_comp_nns = zeros(8,8);

for j = 1:8
    for i = 1:4
        max_value = max([mat_acc_final(2*i-1,j),mat_acc_final(2*i,j)]);
        mat_comp_nns(2*i-1,j) = mat_acc_final(2*i-1,j) - max_value;
        mat_comp_nns(2*i,j) = mat_acc_final(2*i,j) - max_value;
    end
end

cont_best_nns = zeros(1,2);
cont_best_nns(1) = sum(sum(mat_comp_nns(1:2:7,:) == 0));
cont_best_nns(2) = sum(sum(mat_comp_nns(2:2:8,:) == 0));

if (cont_best_nns(1) >= cont_best_nns(2))
    disp('best NN = 1-NN');
else
    disp('best NN = K-NN');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare Kernels

cont_times_best = zeros(1,8);
cont_times_3_best = zeros(1,8);
bests_3_kernels_per_line = zeros(8,3);

for i = 1:8
    line = mat_acc_final(i,:);
    [sorted_line,sorted_indexes] = sort(line,'descend');
    bests_3_kernels_per_line(i,:) = sorted_indexes(1:3);
    cont_times_best(sorted_indexes(1)) = cont_times_best(sorted_indexes(1)) + 1;
    for j = 1:3
        cont_times_3_best(sorted_indexes(j)) = cont_times_3_best(sorted_indexes(j)) + 1;
    end
end

[max_n_times_best,best_kernel_1] = max(cont_times_best);
[max_n_times_3_best,best_kernel_2] = max(cont_times_3_best);

if(best_kernel_2 == 1)
    disp('best kernel = LIN');
elseif(best_kernel_2 == 2)
    disp('best kernel = GAU');
elseif(best_kernel_2 == 3)
    disp('best kernel = POL');
elseif(best_kernel_2 == 4)
    disp('best kernel = EXP');
elseif(best_kernel_2 == 5)
    disp('best kernel = CAU');
elseif(best_kernel_2 == 6)
    disp('best kernel = LOG');
elseif(best_kernel_2 == 7)
    disp('best kernel = SIG');
elseif(best_kernel_2 == 8)
    disp('best kernel = KMOD');
end

%% SPARK - Result Analysis (2) - NN and KNN together

% Set filename

clc;

arquivo = 'spark_motorFailure_02_hold_01.xlsx';
faixa = 'D5:K20';

% Get Complete matrices

planilha = 'acc_mean';

tabela_acc_mean = readtable(arquivo, 'Sheet', planilha, 'Range', faixa,...
                            'ReadVariableNames',false);
mat_acc_mean = table2array(tabela_acc_mean(:,:)); % mat_acc_mean = tabela{:,:};

planilha = 'acc_std';

tabela_acc_std = readtable(arquivo, 'Sheet', planilha, 'Range', faixa,...
                            'ReadVariableNames',false);
mat_acc_std = table2array(tabela_acc_std(:,:)); % mat_acc_mean = tabela{:,:};

planilha = 'K_best';

tabela_K_best = readtable(arquivo, 'Sheet', planilha, 'Range', faixa,...
                            'ReadVariableNames',false);
mat_K_best = table2array(tabela_K_best(:,:)); % mat_acc_mean = tabela{:,:};

% Init Reduced matrices

mat_acc_mean_red = zeros(8,8);
mat_acc_std_red = zeros(8,8);
mat_K_best_red = zeros(8,8);

% Get Reduced Matrices

for i = 1:8
    for j = 1:8
        if(mat_acc_mean(2*i,j) >= mat_acc_mean(2*i-1,j))
            mat_acc_mean_red(i,j) = mat_acc_mean(2*i,j);
            mat_acc_std_red(i,j) = mat_acc_std(2*i,j);
            mat_K_best_red(i,j) = mat_K_best(2*i,j);
        else
            mat_acc_mean_red(i,j) = mat_acc_mean(2*i-1,j);
            mat_acc_std_red(i,j) = mat_acc_std(2*i-1,j);
            mat_K_best_red(i,j) = mat_K_best(2*i-1,j);
        end
    end
end

% %%%%%%%%%%%%%%%%%%%%%%% Max and Min Values

disp('max value: ')
disp(max(max(mat_acc_mean_red)));
disp('min value: ')
disp(min(min(mat_acc_mean_red)));

% %%%%%%%%%%%%%%%%%%%%%%% Get best combination

[max_mean_value,max_mean_index] = max(mat_acc_mean_red(:));
[max_row, max_col] = ind2sub(size(mat_acc_mean_red),max_mean_index);

disp(mat_acc_mean_red == max_mean_value);
disp(sum(sum(mat_acc_mean_red == max_mean_value)))
disp(max_mean_value);

best_result_str1 = 'best result =';

if(max_row <= 2)
    best_result_str2 = ' ALD,';
elseif(max_row <= 4)
    best_result_str2 = ' COH,';
elseif(max_row <= 6)
    best_result_str2 = ' NOV,';
elseif(max_row <= 8)
    best_result_str2 = ' SUR,';
end

if(mod(max_row,2) == 1)
    best_result_str4 = ' DM1,';
else
    best_result_str4 = ' DM2,';
end

if(max_col == 1)
    best_result_str5 = ' LIN';
elseif(max_col == 2)
    best_result_str5 = ' GAU';
elseif(max_col == 3)
    best_result_str5 = ' POL';
elseif(max_col == 4)
    best_result_str5 = ' EXP';
elseif(max_col == 5)
    best_result_str5 = ' CAU';
elseif(max_col == 6)
    best_result_str5 = ' LOG';
elseif(max_col == 7)
    best_result_str5 = ' SIG';
elseif(max_col == 8)
    best_result_str5 = ' KMO';
end

best_result_str = strcat(best_result_str1,best_result_str2,...
                         best_result_str4,best_result_str5);
                        
disp(best_result_str);

% %%%%%%%%%%%%%%%%%%%%%%% Compare Sparsification Methods

mat_acc_comp_sparse = zeros(8,8);
cont_best_sparse = zeros(1,4);

for j = 1:8
    for i = 1:2
        max_value = max([mat_acc_mean_red(i,j),...
                         mat_acc_mean_red(i+2,j),...
                         mat_acc_mean_red(i+4,j),...
                         mat_acc_mean_red(i+6,j)]);
        mat_acc_comp_sparse(i,j) = mat_acc_mean_red(i,j) - max_value;
        mat_acc_comp_sparse(i+2,j) = mat_acc_mean_red(i+2,j) - max_value;
        mat_acc_comp_sparse(i+4,j) = mat_acc_mean_red(i+4,j) - max_value;
        mat_acc_comp_sparse(i+6,j) = mat_acc_mean_red(i+6,j) - max_value;
    end
end

cont_best_sparse(1) = sum(sum( mat_acc_comp_sparse(1:2,:) == 0 ));
cont_best_sparse(2) = sum(sum( mat_acc_comp_sparse(3:4,:) == 0 ));
cont_best_sparse(3) = sum(sum( mat_acc_comp_sparse(5:6,:) == 0 ));
cont_best_sparse(4) = sum(sum( mat_acc_comp_sparse(7:8,:) == 0 ));

if (cont_best_sparse(1) == max(cont_best_sparse))
    disp('best sparsification = ALD');
elseif (cont_best_sparse(2) == max(cont_best_sparse))
    disp('best sparsification = COH');
elseif (cont_best_sparse(3) == max(cont_best_sparse))
    disp('best sparsification = NOV');
elseif (cont_best_sparse(4) == max(cont_best_sparse))
    disp('best sparsification = SURP');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare Design Methods

mat_acc_comp_dms = zeros(8,8);

for j = 1:8
    for i = 1:4
        max_value = max([mat_acc_mean_red(2*i-1,j),mat_acc_mean_red(2*i,j)]);
        mat_acc_comp_dms(2*i-1,j) = mat_acc_mean_red(2*i-1,j) - max_value;
        mat_acc_comp_dms(2*i,j) = mat_acc_mean_red(2*i,j) - max_value;
    end
end

cont_best_dms = zeros(1,2);
cont_best_dms(1) = sum(sum(mat_acc_comp_dms([1,3,5,7],:) == 0));
cont_best_dms(2) = sum(sum(mat_acc_comp_dms([2,4,6,8],:) == 0));

if (cont_best_dms(1) >= cont_best_dms(2))
    disp('best DM = 1');
else
    disp('best DM = 2');
end

% %%%%%%%%%%%%%%%%%%%%%%% Compare Kernels

cont_times_best = zeros(1,8);
cont_times_3_best = zeros(1,8);
bests_3_kernels_per_line = zeros(8,3);

for i = 1:8
    line = mat_acc_mean_red(i,:);
    [sorted_line,sorted_indexes] = sort(line,'descend');
    bests_3_kernels_per_line(i,:) = sorted_indexes(1:3);
    cont_times_best(sorted_indexes(1)) = cont_times_best(sorted_indexes(1)) + 1;
    for j = 1:3
        cont_times_3_best(sorted_indexes(j)) = cont_times_3_best(sorted_indexes(j)) + 1;
    end
end

[max_n_times_best,best_kernel_1] = max(cont_times_best);
[max_n_times_3_best,best_kernel_2] = max(cont_times_3_best);

if(best_kernel_2 == 1)
    disp('best kernel = LIN');
elseif(best_kernel_2 == 2)
    disp('best kernel = GAU');
elseif(best_kernel_2 == 3)
    disp('best kernel = POL');
elseif(best_kernel_2 == 4)
    disp('best kernel = EXP');
elseif(best_kernel_2 == 5)
    disp('best kernel = CAU');
elseif(best_kernel_2 == 6)
    disp('best kernel = LOG');
elseif(best_kernel_2 == 7)
    disp('best kernel = SIG');
elseif(best_kernel_2 == 8)
    disp('best kernel = KMOD');
end

%% END




















