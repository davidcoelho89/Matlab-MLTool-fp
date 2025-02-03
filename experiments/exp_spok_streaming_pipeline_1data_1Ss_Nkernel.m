function [] = exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT, ...
                                                            HP_gs, ...
                                                            PSp, ...
                                                            kernels)

% --- Pipeline used to test spok model with 1 dataset and N Kernels ---
%
%   [] = exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HPgs,PSp,Kernels)
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

if(strcmp(OPT.hpo,'none'))
    hpo_str = '0';
else
    hpo_str = '1';
end

str1 = DATA.name;
str1_1 = '_';
str1_2 = int2str(OPT.prob2);
str2 = '_spok_hold_';
str2_2 = OPT.hold;
str2_3 = '_norm_';
str3 = int2str(OPT.norm);
str3_2 = '_hpo_';
str3_3 = hpo_str;
str4 = '_Dm_';
str5 = int2str(HP_gs.Dm);
str6 = '_Ss_';
str7 = int2str(HP_gs.Ss);
str8 = '_Us_';
str9 = int2str(HP_gs.Us);
str10 = '_Ps_';
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

    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);
    
    disp("finished linear kernel!");
end

%% KERNEL = GAUSSIAN

if (any(kernels == 2))
    
    str12 = '_gau_';
    HP_gs.Ktype = 2;
    
    if(HP_gs.Ss == 1)        % ALD
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
    
    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

    disp("finished gaussian kernel!");

end

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

    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

    disp("finished polynomial kernel!");

end

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

    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

    disp("finished exponential kernel!");

end

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

    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

    disp("finished cauchy kernel!");

end

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

    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

    disp("finished log kernel!");

end

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

    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

    disp("finished sigmoid kernel!");

end

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
    
    HP_gs.sigma = 2.^linspace(-8,2,11);
    HP_gs.gamma = 2.^linspace(-8,2,11);
    HP_gs.alpha = 1;
    HP_gs.theta = 1;

    OPT.file = strcat(str1,str1_1,str1_2,str2,str2_2,str2_3,str3, ...
                      str3_2,str3_3,str4,str5,str6,str7,str8,...
                      str9,str10,str11,str12,str13,str14,str15);

    exp_spok_streaming_pipeline_1data_1Ss_1kernel(DATA,OPT,HP_gs,PSp);

    disp("finished kmod kernel!");

end

%% END