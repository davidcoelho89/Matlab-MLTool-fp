function [] = exp_spok_stationary_pipeline_1data_1Ss_Nkernel_best(OPT,kernels)

% --- Pipeline used to test spok model with 1 dataset and 1 Kernel ---
%
%   [] = exp_spok_stationary_pipeline_1data_1Ss_Nkernel_best(OPT,kernels)
%
%   Input:
%       OPT.
%           prob = which dataset will be used
%           prob2 = a specification of the dataset
%           norm = which normalization will be used
%           lbl = which labeling strategy will be used

%       kernels = list of kernels to be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA LOADING

DATA = data_class_loading(OPT);

%% FILE NAME - STRINGS

str1 = DATA.name;
str1_1 = '_';
str1_2 = int2str(OPT.prob2);
str2 = '_spok_hold_';
str2_2 = int2str(OPT.hold);
str2_3 = '_norm_';
str3 = int2str(OPT.norm);
str3_2 = '_hpo_';
% str3_3 = '1' or 'b';
str4 = '_Dm';
str5 = int2str(OPT.Dm);
str6 = '_Ss';
str7 = int2str(OPT.Ss);
str8 = '_Us';
str9 = int2str(OPT.Us);
str10 = '_Ps';
str11 = int2str(OPT.Ps);
% str12 = '_<kernel>_';
str13 = 'nn';
str14 = int2str(OPT.K);
str15 = '.mat';

%% RUN FILE WITH 1 KERNEL

if (any(kernels == 1))
    str12 = '_lin_';
    OPT.Ktype = 1;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
                  
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished linear kernel!");
end

if (any(kernels == 2))
    str12 = '_gau_';
    OPT.Ktype = 2;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
    
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished gaussian kernel!")
end

if (any(kernels == 3))
    str12 = '_pol_';
    OPT.Ktype = 3;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
    
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished polynomial kernel!");
end

if (any(kernels == 4))
    str12 = '_exp_';
    OPT.Ktype = 4;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
    
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished exponential kernel!");
end

if (any(kernels == 5))
    str12 = '_cau_';
    OPT.Ktype = 5;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
    
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished cauchy kernel!");
end

if (any(kernels == 6))
    str12 = '_log_';
    OPT.Ktype = 6;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
    
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished log kernel!");
end

if (any(kernels == 7))
    str12 = '_sig_';
    OPT.Ktype = 7;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
    
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished sigmoid kernel!");
end

if (any(kernels == 8))
    str12 = '_kmod_';
    OPT.Ktype = 8;
    
    OPT.file_hp = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                         str3_2,'1',str4,str5,str6,str7,str8,str9, ...
                         str10,str11,str12,str13,str14,str15);
                  
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,'b',str4,str5,str6,str7,str8,str9, ...
                      str10,str11,str12,str13,str14,str15);
    
    exp_spok_stationary_pipeline_1data_1Ss_1kernel_best(DATA,OPT);
    
    disp("finished kmod kernel!");
end

%% END