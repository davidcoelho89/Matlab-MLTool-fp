function [] = exp_ksomef_pipeline_1_data_1_lbl_N_kernel(OPT,HP_gs,...
                                                        MP,kernels)

% --- Pipeline used to test spok model with 1 dataset and 1 Kernel ---
%
%   [] = exp_ksomef_pipeline_1_data_1_lbl_N_kernel(OPT,HP_gs,...
%                                                  MP,kernels)
%
%   Input:
%       OPT.
%
%       CVp.
%
%       HP_gs = default hyperparameters
%       kernels = list of kernels to be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA LOADING

DATA = data_class_loading(OPT);

%% FILE NAME - STRINGS

if(strcmp(OPT.hpo,'none'))
    hpo_str = '0';
else
    hpo_str = '1';
end

str1 = DATA.name;
str1_1 = '_';
str1_2 = int2str(OPT.prob2);
str2 = '_ksomef';
str3 = '_hold_';
str3_1 = int2str(OPT.hold);
str4 = '_hpo_';
str4_1 = hpo_str;
str5 = '_norm_';
str5_1 = int2str(OPT.norm);
str6 = '_lbl_';
str6_1 = int2str(HP_gs.lbl);
str7 = '_nn_';
str7_1 = int2str(HP_gs.K);
str8 = '_Nep_';
str8_1 = int2str(HP_gs.Nep);
str9 = '_Nprot_';
str9_1 = int2str(prod(HP_gs.Nk));
str10 = '_Kt_';
% str10_1 = int2str(HP_gs.Ktype);
str11 = '.mat';

%% RUN FILE WITH 1 KERNEL

if (any(kernels == 1))
    
    str10_1 = int2str(1);
    
    HP_gs.Ktype = 1;

    HP_gs.sigma = 2;
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = [0,2.^linspace(-10,10,21)];
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished linear kernel!");
end

if (any(kernels == 2))
    
    str10_1 = int2str(2);
    
    HP_gs.Ktype = 2;
    
    HP_gs.sigma = 2.^linspace(-10,10,21);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished gaussian kernel!");
end

if (any(kernels == 3))
    
    str10_1 = int2str(3);
    
    HP_gs.Ktype = 3;
    
    HP_gs.sigma = 2;
    HP_gs.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
	HP_gs.alpha = 2.^linspace(-10,10,21);
	HP_gs.theta = [0,2.^linspace(-10,10,21)];
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished polynomial kernel!");
end

if (any(kernels == 4))
    
    str10_1 = int2str(4);
    
    HP_gs.Ktype = 4;
    
	HP_gs.sigma = 2.^linspace(-10,10,21);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished exponential kernel!");
end

if (any(kernels == 5))
    
    str10_1 = int2str(5);
    
    HP_gs.Ktype = 5;
    
    HP_gs.sigma = 2.^linspace(-10,10,21);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished cauchy kernel!");
end

if (any(kernels == 6))
    
    str10_1 = int2str(6);
    
    HP_gs.Ktype = 6;
    
    HP_gs.sigma = 2.^linspace(-10,10,21);
	HP_gs.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
    HP_gs.alpha = 1;
    HP_gs.theta = 1;
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished log kernel!");
end

if (any(kernels == 7))
    
    str10_1 = int2str(7);
    
    HP_gs.Ktype = 7;
    
    HP_gs.sigma = 2;
    HP_gs.gamma = 2;
    HP_gs.alpha = 2.^linspace(-10,10,21);
	HP_gs.theta = [-2.^linspace(10,-10,21), 2.^linspace(-10,10,21)];
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished sigmoid kernel!");
end

if (any(kernels == 8))
    
    str10_1 = int2str(8);
    
    HP_gs.Ktype = 8;
    
    HP_gs.sigma = 2.^linspace(-10,10,21);
	HP_gs.gamma = 2.^linspace(-10,10,21);
    HP_gs.alpha = 1;
    HP_gs.theta = 1;
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str3,str3_1,str4,...
                      str4_1,str5,str5_1,str6,str6_1,str7,str7_1, ...
                      str8,str8_1,str9,str9_1,str10,str10_1,str11);
    
    exp_ksomef_pipeline_1_data_1_lbl_1_kernel(DATA,OPT,HP_gs,MP);
    
    disp("finished kmod kernel!");
end




















