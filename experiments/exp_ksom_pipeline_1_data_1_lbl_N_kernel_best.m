function [] = exp_ksom_pipeline_1_data_1_lbl_N_kernel_best(OPT,kernels)

% --- Pipeline used to test ksom model with 1 dataset and 1 Kernel ---
%
%   [] = exp_ksom_pipeline_1_data_1_lbl_N_kernel_best(OPT,kernels)
%
%   Input:
%       OPT.
%           Nr = number of repetitions          [cte]
%           ...
%       kernels = list of kernels to be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA LOADING & ALGORITHM CHOOSING

DATA = data_class_loading(OPT);

if(strcmp(OPT.alg,'ksom_ef'))
    algorithm = 'ksomef';
else
    algorithm = 'ksomgd';
end

%% FILE NAME - STRINGS

str1 = DATA.name;
str1_1 = '_';
str1_2 = int2str(OPT.prob2);
str2_1 = '_';
str2_2 = algorithm;
str3 = '_hold_';
str3_1 = int2str(OPT.hold);
str4 = '_norm_';
str4_1 = int2str(OPT.norm);
str5 = '_hpo_';
% str5_1 =  '1' or 'b';
str6 = '_lbl_';
str6_1 = int2str(OPT.prot_lbl);
str7 = '_nn_';
str7_1 = OPT.nn;
str8 = '_Nep_';
str8_1 = OPT.Nep;
str9 = '_Nprot_';
str9_1 = OPT.Nprot;
str10 = '_Kt_';
% str10_1 = int2str(OPT.Ktype);
str11 = '.mat';

%% RUN FILE WITH 1 KERNEL

if (any(kernels == 1))

str10_1 = int2str(1);

OPT.Ktype = 1;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished linear kernel!");

end

if (any(kernels == 2))

str10_1 = int2str(2);

OPT.Ktype = 2;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished gaussian kernel!");

end

if (any(kernels == 3))

str10_1 = int2str(3);

OPT.Ktype = 3;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished polynomial kernel!");

end

if (any(kernels == 4))

str10_1 = int2str(4);

OPT.Ktype = 4;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished exponetial kernel!");

end

if (any(kernels == 5))

str10_1 = int2str(5);

OPT.Ktype = 5;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished cauchy kernel!");

end

if (any(kernels == 6))

str10_1 = int2str(6);

OPT.Ktype = 6;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished log kernel!");

end

if (any(kernels == 7))

str10_1 = int2str(7);

OPT.Ktype = 7;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished sigmoid kernel!");

end

if (any(kernels == 8))

str10_1 = int2str(8);

OPT.Ktype = 8;

OPT.file_hp = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                     str4_1,str5,'1',str6,str6_1,str7,str7_1, ...
                     str8,str8_1,str9,str9_1,str10,str10_1,str11);
                  
OPT.file = strcat(str1,str1_1,str1_2,str2_1,str2_2,str3,str3_1,str4,...
                  str4_1,str5,'b',str6,str6_1,str7,str7_1, ...
                  str8,str8_1,str9,str9_1,str10,str10_1,str11);

exp_ksom_pipeline_1_data_1_lbl_1_kernel_best(DATA,OPT);
    
disp("finished kmod kernel!");

end

%% END