function [] = exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,...
                                                              PSp,kernels)

% --- Pipeline used to test spok model with 1 dataset and N Kernels ---
%
%   [] = exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HPgs,PSp)
%
%   Input:
%       OPT.
%           prob = which dataset will be used
%           prob2 = a specification of the dataset
%           norm = which normalization will be used
%           lbl = which labeling strategy will be used
%       HP_gs = default hyperparameters
%       PSp.
%           iterations = number of times data is presented to the algorithm
%           type = Takes into account also the dicitionary size
%           lambda = Jpbc = Ds + lambda * Err
%       kernels = list of kernels to be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA LOADING

DATA = data_class_loading(OPT);

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

if (any(kernels == 1))

    str12 = '_lin_';
    
    if(HP_gs.Ss == 1)
        HP_gs.v1 = 2.^linspace(-10,10,21);                  % ALD
    elseif(HP_gs.Ss == 2)
        HP_gs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];   % Coherence
    end
    
    % ToDo - The same thing of v1, to v2! Novelty and Surprise!
    HP_gs.v2 = HP_gs.v1(end) + 0.001;

    HP_gs.Ktype = 1;
    HP_gs.sigma = 2;
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% KERNEL = GAUSSIAN

if (any(kernels == 2))
    
    str12 = '_gau_';
    
    if(HP_gs.Ss == 1)
        HP_gs.v1 = 2.^linspace(-4,3,8);                     % ALD
    elseif(HP_gs.Ss == 2)
        HP_gs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];	% Coherence
    end

    % ToDo - The same thing of v1, to v2! Novelty and Surprise!
    HP_gs.v2 = HP_gs.v1(end) + 0.001;
    
    HP_gs.Ktype = 2;
    HP_gs.sigma = 2.^linspace(-10,9,20);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;
    
    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% KERNEL = POLYNOMIAL

if (any(kernels == 3))

    str12 = '_pol_';

    HP_gs.v1 = 2.^linspace(-13,6,20);
    HP_gs.v2 = HP_gs.v1(end) + 0.001;
    HP_gs.Ktype = 3;
    HP_gs.sigma = 2;
    HP_gs.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% KERNEL = EXPONENTIAL

if (any(kernels == 4))

    str12 = '_exp_';

    HP_gs.v1 = 2.^linspace(-4,3,8);
    HP_gs.v2 = HP_gs.v1(end) + 0.001;
    HP_gs.Ktype = 4;
    HP_gs.sigma = 2.^linspace(-10,9,20);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% KERNEL = CAUCHY

if (any(kernels == 5))

    str12 = '_cau_';

    if(HP_gs.Ss == 1)
        HP_gs.v1 = 2.^linspace(-4,3,8);                     % ALD
    elseif(HP_gs.Ss == 2)
        HP_gs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];	% Coherence
    end
    
    HP_gs.v2 = HP_gs.v1(end) + 0.001;
    HP_gs.Ktype = 5;
    HP_gs.sigma = 2.^linspace(-10,9,20);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% KERNEL = LOG

if (any(kernels == 6))

    str12 = '_log_';

    HP_gs.v1 = -2.^linspace(10,2,9);
    HP_gs.v2 = HP_gs.v1(end) + 0.001;
    HP_gs.Ktype = 6;
    HP_gs.sigma = [0.001 0.01 0.1 1 2 5];
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% KERNEL = SIGMOID

if (any(kernels == 7))
    
    str12 = '_sig_';

    HP_gs.v1 = 2.^linspace(-13,6,20);
    HP_gs.v2 = HP_gs.v1(end) + 0.001;
    HP_gs.Ktype = 7;
    HP_gs.sigma = 2;
    HP_gs.gamma = 2;
    HP_gs.alpha = 2.^linspace(-8,2,11);       
    % HP_gs.theta = 2.^linspace(-8,2,11);
    HP_gs.theta = 0.1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% KERNEL = KMOD

if (any(kernels == 8))

    str12 = '_kmod_';

    HP_gs.v1 = 2.^linspace(-13,6,20);
    HP_gs.v2 = HP_gs.v1(end) + 0.001;
    HP_gs.Ktype = 8;
    HP_gs.sigma = 2.^linspace(-8,2,11);
    HP_gs.gamma = 2.^linspace(-8,2,11);
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

end

%% END