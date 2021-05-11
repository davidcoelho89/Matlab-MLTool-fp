function [] = exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT)

% --- Pipeline used to test isk2nn model with 1 dataset and 1 Kernel ---
%
%   [] = exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HPgs,PSp)
%
%   Input:
%       OPT.
%           prob = which dataset will be used
%           prob2 = a specification of the dataset
%           norm = which normalization will be used
%           lbl = which labeling strategy will be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA LOADING

DATA = data_class_loading(OPT);

%% CROSS VALIDATION OPTIONS

PSpar.iterations = 1;	% number of times data is presented to the algorithm
PSpar.type = 2;         % Takes into account also the dicitionary size
PSpar.lambda = 2;       % Jpbc = Ds + lambda * Err

%% HYPERPARAMETERS - DEFAULT

HP_gs.Ne = 01;
HP_gs.Dm = 2;
HP_gs.Ss = 2;
HP_gs.v1 = 0.4;
HP_gs.v2 = 0.9;
HP_gs.Us = 3;
HP_gs.eta = 0.3;
HP_gs.Ps = 2;
HP_gs.min_score = -10;
HP_gs.max_prot = 600;
HP_gs.min_prot = 1;
HP_gs.Von = 0;
HP_gs.K = 1;
HP_gs.knn_type = 1;
HP_gs.Ktype = 1;
HP_gs.sig2n = 0.001;

%% FILE NAME - STRINGS

str1 = DATA.name;
str2 = '_isk2nn_hpo1_norm';
str3 = int2str(OPT.norm);
str4 = '_Dm';
str5 = int2str(HP_gs.Dm);
str6 = '_Ss';
str7 = int2str(HP_gs.Ss);
str8 = '_Us';
str9 = int2str(HP_gs.Us);
str10 = '_Ps';
str11 = int2str(HP_gs.Ps);
% str12 = '_<kernel>_';
str13 = int2str(HP_gs.K);
str14 = 'nn.mat';

%% KERNEL = LINEAR

str12 = '_lin_';

% HP_gs.v1 = 2.^linspace(-10,10,21);                  % ALD
HP_gs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];   % Coherence
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 1;
HP_gs.sigma = 2;
HP_gs.gamma = 2;
HP_gs.alpha = 1;
HP_gs.theta = 1;

OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                  str9,str10,str11,str12,str13,str14);

exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% KERNEL = GAUSSIAN

str12 = '_gau_';

% HP_gs.v1 = 2.^linspace(-4,3,8);                     % ALD
HP_gs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];	% Coherence
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 2;
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;
HP_gs.alpha = 1;
HP_gs.theta = 1;

OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                  str9,str10,str11,str12,str13,str14);


exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% KERNEL = POLYNOMIAL

% str12 = '_pol_';
% 
% HP_gs.v1 = 2.^linspace(-13,6,20);
% HP_gs.v2 = HP_gs.v1(end) + 0.001;
% HP_gs.Ktype = 3;
% HP_gs.sigma = 2;
% HP_gs.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
% HP_gs.alpha = 1;
% HP_gs.theta = 1;
% 
% OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
%                   str9,str10,str11,str12,str13,str14);
% 
% exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% KERNEL = EXPONENTIAL

% str12 = '_exp_';
% 
% HP_gs.v1 = 2.^linspace(-4,3,8);
% HP_gs.v2 = HP_gs.v1(end) + 0.001;
% HP_gs.Ktype = 4;
% HP_gs.sigma = 2.^linspace(-10,9,20);
% HP_gs.gamma = 2;
% HP_gs.alpha = 1;
% HP_gs.theta = 1;
% 
% OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
%                   str9,str10,str11,str12,str13,str14);
% 
% 
% exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% KERNEL = CAUCHY

str12 = '_cau_';

% HP_gs.v1 = 2.^linspace(-4,3,8);                     % ALD
HP_gs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];	% Coherence
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 5;
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;
HP_gs.alpha = 1;
HP_gs.theta = 1;

OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                  str9,str10,str11,str12,str13,str14);

exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% KERNEL = LOG

% str12 = '_log_';
% 
% HP_gs.v1 = -2.^linspace(10,2,9);
% HP_gs.v2 = HP_gs.v1(end) + 0.001;
% HP_gs.Ktype = 6;
% HP_gs.sigma = [0.001 0.01 0.1 1 2 5];
% HP_gs.gamma = 2;
% HP_gs.alpha = 1;
% HP_gs.theta = 1;
% 
% OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
%                   str9,str10,str11,str12,str13,str14);
% 
% exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% KERNEL = SIGMOID

% str12 = '_sig_';
% 
% HP_gs.v1 = 2.^linspace(-13,6,20);
% HP_gs.v2 = HP_gs.v1(end) + 0.001;
% HP_gs.Ktype = 7;
% HP_gs.sigma = 2;
% HP_gs.gamma = 2;
% HP_gs.alpha = 2.^linspace(-8,2,11);       
% % HP_gs.theta = 2.^linspace(-8,2,11);
% HP_gs.theta = 0.1;
% 
% OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
%                   str9,str10,str11,str12,str13,str14);
% 
% exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% KERNEL = KMOD

% str12 = '_kmod_';
% 
% HP_gs.v1 = 2.^linspace(-13,6,20);
% HP_gs.v2 = HP_gs.v1(end) + 0.001;
% HP_gs.Ktype = 8;
% HP_gs.sigma = 2.^linspace(-8,2,11);
% HP_gs.gamma = 2.^linspace(-8,2,11);
% HP_gs.alpha = 1;
% HP_gs.theta = 1;
% 
% OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
%                   str9,str10,str11,str12,str13,str14);
% 
% exp_isk2nn_pipeline_streaming_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSpar);

%% END