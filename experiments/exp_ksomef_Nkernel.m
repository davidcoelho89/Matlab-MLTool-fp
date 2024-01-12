%% Machine Learning ToolBox

% KSOM-EF Comparison with N Kernels
% Author: David Nascimento Coelho
% Last Update: 2024/01/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 10;        % Number of experiment realizations
OPT.alg = 'ksomef'; % Which Classifier will be used
OPT.prob = 10;      % Which problem will be solved / used
OPT.prob2 = 01;     % When it needs an specification of data set
OPT.norm = 3;       % Normalization definition
OPT.lbl = 1;        % Data labeling definition
OPT.hold = 01;      % Hold out method
OPT.ptrn = 0.7;     % Percentage of samples for training
OPT.hpo = 'random'; % 'grid' ; 'random' ; 'none'

OPT.savefile = 1;   % decides if file will be saved

OPT.calculate_bin = 0;  % [0 or 1] decides to calculate binary statistics
OPT.class_1_vect = 1;   % [2,3] which classes belongs together
                        % (for binary statistics)

% Prototypes' labeling strategy

prot_lbl = 1;               % = 1 (MV) / 2 (AD) / 3 (MD)

% Metaparameters

MP.max_it = 020;   	% Maximum number of iterations (random search)
MP.fold = 5;     	% number of data partitions (cross validation)
MP.cost = 2;        % Takes into account also the dicitionary size
MP.lambda = 2.0;    % Jpbc = Ds + lambda * Err

%% CHOOSE FIXED HYPERPARAMETERS 

HP_common.lbl = prot_lbl;       % Neurons' labeling function;
HP_common.ep = 50;              % max number of epochs
HP_common.lin.k = [5 4];        % number of neurons (prototypes)
HP_common.init = 02;            % neurons' initialization
HP_common.dist = 02;            % type of distance
HP_common.learn = 02;           % type of learning step
HP_common.No = 0.7;             % initial learning step
HP_common.Nt = 0.01;            % final learning step
HP_common.Nn = 01;              % number of neighbors
HP_common.neig = 03;            % type of neighbor function
HP_common.Vo = 0.8;             % initial neighbor constant
HP_common.Vt = 0.3;             % final neighbor constant
HP_common.Von = 0;              % disable video
HP_common.K = 1;                % Number of nearest neighbors (classify)
HP_common.knn_type = 2;         % Type of knn aproximation

HP_ksomef_lin = HP_common;      % Get common HP
HP_ksomef_lin.Ktype = 1;        % Type of Kernel
HP_ksomef_lin.theta = 0;     	% Dot product adding

HP_ksomef_gau = HP_common;      % Get common HP
HP_ksomef_gau.Ktype = 2;        % Type of Kernel
HP_ksomef_gau.sigma = 0.5;     	% Kernel width

HP_ksomef_pol = HP_common;      % Get common HP
HP_ksomef_pol.Ktype = 3;        % Type of Kernel
HP_ksomef_pol.alpha = 1;    	% Dot product multiplier
HP_ksomef_pol.theta = 1;     	% Dot product adding
HP_ksomef_pol.gamma = 2;       	% Polynomial order

HP_ksomef_exp = HP_common;      % Get common HP
HP_ksomef_exp.Ktype = 4;        % Type of Kernel
HP_ksomef_exp.sigma = 0.5;     	% Kernel width

HP_ksomef_cau = HP_common;      % Get common HP
HP_ksomef_cau.Ktype = 5;        % Type of Kernel
HP_ksomef_cau.sigma = 0.5;     	% Kernel width

HP_ksomef_log = HP_common;      % Get common HP
HP_ksomef_log.Ktype = 6;        % Type of Kernel
HP_ksomef_log.sigma = 2;     	% Kernel width
HP_ksomef_log.gamma = 2;       	% Exponential order

HP_ksomef_sig = HP_common;      % Get common HP
HP_ksomef_sig.Ktype = 7;        % Type of Kernel
HP_ksomef_sig.alpha = 0.1;    	% Dot product multiplier
HP_ksomef_sig.theta = 0.1;     	% Dot product adding

HP_ksomef_kmo = HP_common;      % Get common HP
HP_ksomef_kmo.Ktype = 8;        % Type of Kernel
HP_ksomef_kmo.sigma = 2;     	% Kernel width
HP_ksomef_kmo.gamma = 2;       	% Exponential order

%% HYPERPARAMETERS - FOR OPTIMIZATION

if(~strcmp(OPT.hpo,'none'))

HPgs_ksomef_lin = HP_ksomef_lin;
HPgs_ksomef_lin.theta = [0,2.^linspace(-10,10,21)];

HPgs_ksomef_gau = HP_ksomef_gau;
HPgs_ksomef_gau.sigma = 2.^linspace(-10,10,21);

HPgs_ksomef_pol = HP_ksomef_pol;
HPgs_ksomef_pol.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
HPgs_ksomef_pol.alpha = 2.^linspace(-10,10,21);
HPgs_ksomef_pol.theta = [0,2.^linspace(-10,10,21)];

HPgs_ksomef_exp = HP_ksomef_exp;
HPgs_ksomef_exp.sigma = 2.^linspace(-10,10,21);

HPgs_ksomef_cau = HP_ksomef_cau;
HPgs_ksomef_cau.sigma = 2.^linspace(-10,10,21);

HPgs_ksomef_log = HP_ksomef_log;
HPgs_ksomef_log.sigma = 2.^linspace(-10,10,21);
HPgs_ksomef_log.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];

HPgs_ksomef_sig = HP_ksomef_sig;
HPgs_ksomef_sig.alpha = 2.^linspace(-10,10,21);
HPgs_ksomef_sig.theta = [-2.^linspace(10,-10,21), 2.^linspace(-10,10,21)];

HPgs_ksomef_kmo = HP_ksomef_kmo;
HPgs_ksomef_kmo.sigma = 2.^linspace(-10,10,21);
HPgs_ksomef_kmo.gamma = 2.^linspace(-10,10,21);

end

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);         % See pairplot of attributes

[Nc,~] = size(DATA.output);         % Get number of classes

%% ACCUMULATORS AND HANDLERS

% Acc of labels and data division
data_acc = cell(OPT.Nr,1);         	

% Acc of names for plots
NAMES = {'Linear','Gaussian',...    
         'Polynomial', 'Exponential',...
         'Cauchy', 'Log',...
         'Sigmoid', 'Kmod'};    

nstats_all_tr = cell(length(NAMES),1);
nstats_all_ts = cell(length(NAMES),1);

ksomef_lin_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_lin_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_lin_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_lin_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_lin_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomef_gau_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_gau_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_gau_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_gau_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_gau_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomef_pol_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_pol_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_pol_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_pol_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_pol_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomef_exp_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_exp_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_exp_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_exp_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_exp_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomef_cau_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_cau_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_cau_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_cau_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_cau_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomef_log_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_log_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_log_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_log_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_log_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomef_sig_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_sig_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_sig_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_sig_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_sig_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomef_kmo_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_kmo_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_kmo_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_kmo_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_kmo_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

%% FILE NAME

OPT.filename = strcat(DATA.name,'_prob2_',int2str(OPT.prob2),'_ksomef',...
                      '_hpo_',OPT.hpo,'_norm',int2str(OPT.norm),'_1nn');

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp(r);
display(datetime("now"));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

[DATAho] = hold_out(DATA,OPT);

data_acc{r} = DATAho;
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

% %%%%%%%%%%%%%%%%% NORMALIZE DATA %%%%%%%%%%%%%%%%%%%%%%%

% Get Normalization Parameters

PARnorm = normalize_fit(DATAtr,OPT);

% Training data normalization

DATAtr = normalize_transform(DATAtr,PARnorm);

% Test data normalization

DATAts = normalize_transform(DATAts,PARnorm);

% Adjust Values for video function

DATA = normalize_transform(DATA,PARnorm);
DATAtr.Xmax = max(DATA.input,[],2);  % max value
DATAtr.Xmin = min(DATA.input,[],2);  % min value
DATAtr.Xmed = mean(DATA.input,2);    % mean value
DATAtr.Xdp = std(DATA.input,[],2);   % std value

% %%%%%%%%%%%%%% SHUFFLE TRAINING DATA %%%%%%%%%%%%%%%%%%%

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);
DATAtr.lbl = DATAtr.lbl(:,I);

% %%%%%%%%%%% HYPERPARAMETER OPTIMIZATION %%%%%%%%%%%%%%%%

if(strcmp(OPT.hpo,'none'))
    % Does nothing
elseif(strcmp(OPT.hpo,'random'))
    HP_ksomef_lin = random_search_cv(DATAtr,HPgs_ksomef_lin,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
    HP_ksomef_gau = random_search_cv(DATAtr,HPgs_ksomef_gau,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
    HP_ksomef_pol = random_search_cv(DATAtr,HPgs_ksomef_pol,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
    HP_ksomef_exp = random_search_cv(DATAtr,HPgs_ksomef_exp,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
    HP_ksomef_cau = random_search_cv(DATAtr,HPgs_ksomef_cau,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
    HP_ksomef_log = random_search_cv(DATAtr,HPgs_ksomef_log,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
    HP_ksomef_sig = random_search_cv(DATAtr,HPgs_ksomef_sig,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
    HP_ksomef_kmo = random_search_cv(DATAtr,HPgs_ksomef_kmo,...
                                     @ksom_ef_train,@ksom_ef_classify,MP);
end

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

ksomef_lin_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_lin);
ksomef_gau_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_gau);
ksomef_pol_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_pol);
ksomef_exp_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_exp);
ksomef_cau_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_cau);
ksomef_log_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_log);
ksomef_sig_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_sig);
ksomef_kmo_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef_kmo);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

ksomef_lin_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_lin_par_acc{r});
ksomef_lin_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_lin_out_tr_acc{r});
ksomef_lin_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_lin_par_acc{r});
ksomef_lin_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_lin_out_ts_acc{r});

ksomef_gau_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_gau_par_acc{r});
ksomef_gau_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_gau_out_tr_acc{r});
ksomef_gau_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_gau_par_acc{r});
ksomef_gau_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_gau_out_ts_acc{r});

ksomef_pol_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_pol_par_acc{r});
ksomef_pol_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_pol_out_tr_acc{r});
ksomef_pol_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_pol_par_acc{r});
ksomef_pol_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_pol_out_ts_acc{r});

ksomef_exp_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_exp_par_acc{r});
ksomef_exp_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_exp_out_tr_acc{r});
ksomef_exp_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_exp_par_acc{r});
ksomef_exp_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_exp_out_ts_acc{r});

ksomef_cau_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_cau_par_acc{r});
ksomef_cau_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_cau_out_tr_acc{r});
ksomef_cau_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_cau_par_acc{r});
ksomef_cau_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_cau_out_ts_acc{r});

ksomef_log_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_log_par_acc{r});
ksomef_log_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_log_out_tr_acc{r});
ksomef_log_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_log_par_acc{r});
ksomef_log_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_log_out_ts_acc{r});

ksomef_sig_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_sig_par_acc{r});
ksomef_sig_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_sig_out_tr_acc{r});
ksomef_sig_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_sig_par_acc{r});
ksomef_sig_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_sig_out_ts_acc{r});

ksomef_kmo_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_kmo_par_acc{r});
ksomef_kmo_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_kmo_out_tr_acc{r});
ksomef_kmo_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_kmo_par_acc{r});
ksomef_kmo_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_kmo_out_ts_acc{r});

end

%% RESULTS / STATISTICS

% Statistics for n turns 

ksomef_lin_nstats_tr = class_stats_nturns(ksomef_lin_stats_tr_acc);
ksomef_lin_nstats_ts = class_stats_nturns(ksomef_lin_stats_ts_acc);

ksomef_gau_nstats_tr = class_stats_nturns(ksomef_gau_stats_tr_acc);
ksomef_gau_nstats_ts = class_stats_nturns(ksomef_gau_stats_ts_acc);

ksomef_pol_nstats_tr = class_stats_nturns(ksomef_pol_stats_tr_acc);
ksomef_pol_nstats_ts = class_stats_nturns(ksomef_pol_stats_ts_acc);

ksomef_exp_nstats_tr = class_stats_nturns(ksomef_exp_stats_tr_acc);
ksomef_exp_nstats_ts = class_stats_nturns(ksomef_exp_stats_ts_acc);

ksomef_cau_nstats_tr = class_stats_nturns(ksomef_cau_stats_tr_acc);
ksomef_cau_nstats_ts = class_stats_nturns(ksomef_cau_stats_ts_acc);

ksomef_log_nstats_tr = class_stats_nturns(ksomef_log_stats_tr_acc);
ksomef_log_nstats_ts = class_stats_nturns(ksomef_log_stats_ts_acc);

ksomef_sig_nstats_tr = class_stats_nturns(ksomef_sig_stats_tr_acc);
ksomef_sig_nstats_ts = class_stats_nturns(ksomef_sig_stats_ts_acc);

ksomef_kmo_nstats_tr = class_stats_nturns(ksomef_kmo_stats_tr_acc);
ksomef_kmo_nstats_ts = class_stats_nturns(ksomef_kmo_stats_ts_acc);

% Statistics for n turns (binary)

if(OPT.calculate_bin == 1)

ksomef_lin_stats_tr_bin_acc = calculate_binary_stats(ksomef_lin_stats_tr_acc,OPT.class_1_vect);
ksomef_lin_stats_ts_bin_acc = calculate_binary_stats(ksomef_lin_stats_ts_acc,OPT.class_1_vect);
ksomef_lin_nstats_tr_bin = class_stats_nturns(ksomef_lin_stats_tr_bin_acc);
ksomef_lin_nstats_ts_bin = class_stats_nturns(ksomef_lin_stats_ts_bin_acc);

ksomef_gau_stats_tr_bin_acc = calculate_binary_stats(ksomef_gau_stats_tr_acc,OPT.class_1_vect);
ksomef_gau_stats_ts_bin_acc = calculate_binary_stats(ksomef_gau_stats_ts_acc,OPT.class_1_vect);
ksomef_gau_nstats_tr_bin = class_stats_nturns(ksomef_gau_stats_tr_bin_acc);
ksomef_gau_nstats_ts_bin = class_stats_nturns(ksomef_gau_stats_ts_bin_acc);

ksomef_pol_stats_tr_bin_acc = calculate_binary_stats(ksomef_pol_stats_tr_acc,OPT.class_1_vect);
ksomef_pol_stats_ts_bin_acc = calculate_binary_stats(ksomef_pol_stats_ts_acc,OPT.class_1_vect);
ksomef_pol_nstats_tr_bin = class_stats_nturns(ksomef_pol_stats_tr_bin_acc);
ksomef_pol_nstats_ts_bin = class_stats_nturns(ksomef_pol_stats_ts_bin_acc);

ksomef_exp_stats_tr_bin_acc = calculate_binary_stats(ksomef_exp_stats_tr_acc,OPT.class_1_vect);
ksomef_exp_stats_ts_bin_acc = calculate_binary_stats(ksomef_exp_stats_ts_acc,OPT.class_1_vect);
ksomef_exp_nstats_tr_bin = class_stats_nturns(ksomef_exp_stats_tr_bin_acc);
ksomef_exp_nstats_ts_bin = class_stats_nturns(ksomef_exp_stats_ts_bin_acc);

ksomef_cau_stats_tr_bin_acc = calculate_binary_stats(ksomef_cau_stats_tr_acc,OPT.class_1_vect);
ksomef_cau_stats_ts_bin_acc = calculate_binary_stats(ksomef_cau_stats_ts_acc,OPT.class_1_vect);
ksomef_cau_nstats_tr_bin = class_stats_nturns(ksomef_cau_stats_tr_bin_acc);
ksomef_cau_nstats_ts_bin = class_stats_nturns(ksomef_cau_stats_ts_bin_acc);

ksomef_log_stats_tr_bin_acc = calculate_binary_stats(ksomef_log_stats_tr_acc,OPT.class_1_vect);
ksomef_log_stats_ts_bin_acc = calculate_binary_stats(ksomef_log_stats_ts_acc,OPT.class_1_vect);
ksomef_log_nstats_tr_bin = class_stats_nturns(ksomef_log_stats_tr_bin_acc);
ksomef_log_nstats_ts_bin = class_stats_nturns(ksomef_log_stats_ts_bin_acc);

ksomef_sig_stats_tr_bin_acc = calculate_binary_stats(ksomef_sig_stats_tr_acc,OPT.class_1_vect);
ksomef_sig_stats_ts_bin_acc = calculate_binary_stats(ksomef_sig_stats_ts_acc,OPT.class_1_vect);
ksomef_sig_nstats_tr_bin = class_stats_nturns(ksomef_sig_stats_tr_bin_acc);
ksomef_sig_nstats_ts_bin = class_stats_nturns(ksomef_sig_stats_ts_bin_acc);

ksomef_kmo_stats_tr_bin_acc = calculate_binary_stats(ksomef_kmo_stats_tr_acc,OPT.class_1_vect);
ksomef_kmo_stats_ts_bin_acc = calculate_binary_stats(ksomef_kmo_stats_ts_acc,OPT.class_1_vect);
ksomef_kmo_nstats_tr_bin = class_stats_nturns(ksomef_kmo_stats_tr_bin_acc);
ksomef_kmo_nstats_ts_bin = class_stats_nturns(ksomef_kmo_stats_ts_bin_acc);

end

% Get all Statistics in one Cell

nstats_all_tr{1,1} = ksomef_lin_nstats_tr;
nstats_all_tr{2,1} = ksomef_gau_nstats_tr;
nstats_all_tr{3,1} = ksomef_pol_nstats_tr;
nstats_all_tr{4,1} = ksomef_exp_nstats_tr;
nstats_all_tr{5,1} = ksomef_cau_nstats_tr;
nstats_all_tr{6,1} = ksomef_log_nstats_tr;
nstats_all_tr{7,1} = ksomef_sig_nstats_tr;
nstats_all_tr{8,1} = ksomef_kmo_nstats_tr;

nstats_all_ts{1,1} = ksomef_lin_nstats_ts;
nstats_all_ts{2,1} = ksomef_gau_nstats_ts;
nstats_all_ts{3,1} = ksomef_pol_nstats_ts;
nstats_all_ts{4,1} = ksomef_exp_nstats_ts;
nstats_all_ts{5,1} = ksomef_cau_nstats_ts;
nstats_all_ts{6,1} = ksomef_log_nstats_ts;
nstats_all_ts{7,1} = ksomef_sig_nstats_ts;
nstats_all_ts{8,1} = ksomef_kmo_nstats_ts;

if(OPT.calculate_bin == 1)

nstats_all_tr_bin{1,1} = ksomef_lin_nstats_tr_bin;
nstats_all_tr_bin{2,1} = ksomef_gau_nstats_tr_bin;
nstats_all_tr_bin{3,1} = ksomef_pol_nstats_tr_bin;
nstats_all_tr_bin{4,1} = ksomef_exp_nstats_tr_bin;
nstats_all_tr_bin{5,1} = ksomef_cau_nstats_tr_bin;
nstats_all_tr_bin{6,1} = ksomef_log_nstats_tr_bin;
nstats_all_tr_bin{7,1} = ksomef_sig_nstats_tr_bin;
nstats_all_tr_bin{8,1} = ksomef_kmo_nstats_tr_bin;

nstats_all_ts_bin{1,1} = ksomef_lin_nstats_ts_bin;
nstats_all_ts_bin{2,1} = ksomef_gau_nstats_ts_bin;
nstats_all_ts_bin{3,1} = ksomef_pol_nstats_ts_bin;
nstats_all_ts_bin{4,1} = ksomef_exp_nstats_ts_bin;
nstats_all_ts_bin{5,1} = ksomef_cau_nstats_ts_bin;
nstats_all_ts_bin{6,1} = ksomef_log_nstats_ts_bin;
nstats_all_ts_bin{7,1} = ksomef_sig_nstats_ts_bin;
nstats_all_ts_bin{8,1} = ksomef_kmo_nstats_ts_bin;

end

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all_tr,NAMES);
class_stats_ncomp(nstats_all_ts,NAMES);

if(OPT.calculate_bin == 1)
    
class_stats_ncomp(nstats_all_tr_bin,NAMES);
class_stats_ncomp(nstats_all_ts_bin,NAMES);

end

%% SAVE DATA

if(OPT.savefile)
    variables.nstats_all_tr = nstats_all_tr;
    variables.nstats_all_ts = nstats_all_ts;
    variables.ksomef_lin_par_acc = ksomef_lin_par_acc;
    variables.ksomef_gau_par_acc = ksomef_gau_par_acc;
    variables.ksomef_pol_par_acc = ksomef_pol_par_acc;
    variables.ksomef_exp_par_acc = ksomef_exp_par_acc;
    variables.ksomef_cau_par_acc = ksomef_cau_par_acc;
    variables.ksomef_log_par_acc = ksomef_log_par_acc;
    variables.ksomef_sig_par_acc = ksomef_sig_par_acc;
    variables.ksomef_kmo_par_acc = ksomef_kmo_par_acc;
    disp(variables);
    save(OPT.filename,'variables');
    clear variables;
end

%% END