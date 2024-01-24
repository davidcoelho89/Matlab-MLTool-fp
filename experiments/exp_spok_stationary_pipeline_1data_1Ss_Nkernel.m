function [] = exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,...
                                                             CVp,kernels)

% --- Pipeline used to test spok model with 1 dataset and 1 Kernel ---
%
%   [] = exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT)
%
%   Input:
%       OPT.
%           prob = which dataset will be used
%           prob2 = a specification of the dataset
%           norm = which normalization will be used
%           lbl = which labeling strategy will be used
%       CVp.
%           max_it = Maximum number of iterations (random search)
%           fold = number of data partitions for cross validation
%           cost = Which cost function will be used
%           lambda = Jpbc = Ds + lambda * Err (prototype-based models)
%       HP_gs = default hyperparameters
%       kernels = list of kernels to be used
%   Output:
%       "Do not have. Just save structures into a file"

%% DATA LOADING

DATA = data_class_loading(OPT);

%% FILE NAME - STRINGS

str1 = DATA.name;
str2 = '_spok_hpo1_norm';
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
str13 = 'nn';
str14 = int2str(HP_gs.K(1));
str15 = '.mat';

%% KERNEL = LINEAR

if (any(kernels == 1))

    str12 = '_lin_';
    HP_gs.Ktype = 1;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
    
    HP_gs.sigma = 2;
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = [0,2.^linspace(-10,10,21)];

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished linear kernel!");

%% KERNEL = GAUSSIAN

if (any(kernels == 2))
    
    str12 = '_gau_';
    HP_gs.Ktype = 2;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
    
    HP_gs.sigma = 2.^linspace(-10,10,21);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished gaussian kernel!");

%% KERNEL = POLYNOMIAL

if (any(kernels == 3))
    
    str12 = '_pol_';
    HP_gs.Ktype = 3;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
    
    HP_gs.sigma = 2;
   	HP_gs.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
	HP_gs.alpha = 2.^linspace(-10,10,21);
   	HP_gs.theta = [0,2.^linspace(-10,10,21)];

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished polynomial kernel!");

%% KERNEL = EXPONENTIAL

if (any(kernels == 4))

    str12 = '_exp_';
    HP_gs.Ktype = 4;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
    
    HP_gs.sigma = 2.^linspace(-10,10,21);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished exponential kernel!");

%% KERNEL = CAUCHY

if (any(kernels == 5))
    
    str12 = '_cau_';
    HP_gs.Ktype = 5;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
        
    HP_gs.sigma = 2.^linspace(-10,9,20);
    HP_gs.gamma = 2;
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished cauchy kernel!");

%% KERNEL = LOG

if (any(kernels == 6))

    str12 = '_log_';
    HP_gs.Ktype = 6;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = [-2.^linspace(10,-10,21), 2.^linspace(-10,10,21)];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = [-2.^linspace(10,-10,21), 2.^linspace(-10,10,21)];
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
    
    HP_gs.sigma = 2.^linspace(-10,10,21);
	HP_gs.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished log kernel!");

%% KERNEL = SIGMOID

if (any(kernels == 7))

    str12 = '_sig_';
    HP_gs.Ktype = 7;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
    
    HP_gs.sigma = 2;
    HP_gs.gamma = 2;
    HP_gs.alpha = 2.^linspace(-10,10,21);
	HP_gs.theta = [-2.^linspace(10,-10,21), 2.^linspace(-10,10,21)];

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished sigmoid kernel!");

%% KERNEL = KMOD

if (any(kernels == 8))

    str12 = '_kmod_';
    HP_gs.Ktype = 8;
    
    if(HP_gs.Ss == 1)       % ALD
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 2)   % Coherence
        HP_gs.v1 = [0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999];
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
    elseif(HP_gs.Ss == 3)   % Novelty
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    elseif(HP_gs.Ss == 4)   % Surprise
        HP_gs.v1 = 2.^linspace(-10,10,21);
        HP_gs.v2 = HP_gs.v1 + 2^(-10);
    end    
    
	HP_gs.sigma = 2.^linspace(-10,10,21);
	HP_gs.gamma = 2.^linspace(-10,10,21);
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str2,str3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_stationary_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,CVp);

end

disp("finished kmod kernel!");

%% END