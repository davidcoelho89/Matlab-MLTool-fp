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
str1_1 = int2str(OPT.prob2);
if(OPT.prob == 7)
    str2 = 'isk2nn_'
else
    str2 = '_spok_hold';
end
str2_2 = int2str(OPT.hold);
str2_3 = '_norm';
str3 = int2str(OPT.norm);
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


