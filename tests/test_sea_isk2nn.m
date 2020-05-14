%% Machine Learning ToolBox

% SEA Concepts and isk2nn classifier (using isk2nn pipeline)
% Author: David Nascimento Coelho
% Last Update: 2020/05/11

format long e;  % Output data style (float)

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

%% OPTIONS AND HYPERPARAMETERS - DEFAULT

% DATASET: 25 (Sea) / NORM: 0 (no norm)/std / LBL: 1 [-1 +1]

DATAopt =  struct('prob',25,'prob2',1,'norm',0,'lbl',1);

% HPO: 2 (accuracy and dictionary size)

PSpar.iterations = 1;  % number of times data is presented to the algorithm
PSpar.type = 2;        % Takes into account also the dicitionary size
PSpar.lambda = 0.5;    % Jpbc = Ds + lambda * Err

% DM: 2 / SS: 1 / US: 1 / PS: 2 / K: 1 (NN)

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 0.8;
HP_gs.v2 = 0.9;
HP_gs.Us = 1;
HP_gs.eta = 0.01;
HP_gs.Ps = 2;
HP_gs.min_score = -10;
HP_gs.max_prot = 600;
HP_gs.min_prot = 1;
HP_gs.Von = 0;
HP_gs.K = 1;
HP_gs.sig2n = 0.001;

%% KERNEL = LINEAR

DATAopt.file = 'sea_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_lin_nn.mat';
          
HP_gs.v1 = 2.^linspace(-10,10,21);
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 1;       
HP_gs.sigma = 2;
HP_gs.gamma = 2;       
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_isk2nn_pipeline(DATAopt,HP_gs,PSpar)

%% KERNEL = GAUSSIAN

DATAopt.file = 'sea_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_gau_nn.mat';

HP_gs.v1 = 2.^linspace(-4,3,8);
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 2;       
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;       
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_isk2nn_pipeline(DATAopt,HP_gs,PSpar);

%% KERNEL = POLYNOMIAL

DATAopt.file = 'sea_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_pol_nn.mat';

HP_gs.v1 = 2.^linspace(-13,6,20);
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 3;
HP_gs.sigma = 2;
HP_gs.gamma = [2,2.2,2.4,2.6,2.8,3];
HP_gs.alpha = 1;
HP_gs.theta = 1;

test_isk2nn_pipeline(DATAopt,HP_gs,PSpar);

%% KERNEL = CAUCHY

DATAopt.file = 'sea_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_cau_nn.mat';

HP_gs.v1 = 2.^linspace(-4,3,8);
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 5;       
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;
HP_gs.alpha = 1;       
HP_gs.theta = 1;  

test_isk2nn_pipeline(DATAopt,HP_gs,PSpar);

%% KERNEL = SIGMOID

DATAopt.file = 'sea_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_sig_nn.mat';

HP_gs.v1 = 2.^linspace(-13,6,20);
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.Ktype = 7;       
HP_gs.sigma = 2;
HP_gs.gamma = 2;
HP_gs.alpha = 2.^linspace(-8,2,11);       
HP_gs.theta = 0.1;       

test_isk2nn_pipeline(DATAopt,HP_gs,PSpar);

%% END