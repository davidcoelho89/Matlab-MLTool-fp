%% Machine Learning ToolBox

% KSOM-GD Comparison Tests
% Author: David Nascimento Coelho
% Last Update: 2023/03/05

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 05;        % Number of repetitions of each algorithm
OPT.alg = 'ksomgd'; % Which Classifier will be used
OPT.prob = 07;      % Which problem will be solved / used
OPT.prob2 = 02;     % When it needs an specification of data set
OPT.norm = 3;       % Normalization definition
OPT.lbl = 1;        % Data labeling definition
OPT.hold = 02;      % Hold out method
OPT.ptrn = 0.7;     % Percentage of samples for training
OPT.hpo = 'random'; % 'grid' ; 'random' ; 'none'

OPT.savefile = 1;   % decides if file will be saved

OPT.calculate_bin = 0;  % [0 or 1] decides to calculate binary statistics
OPT.class_1_vect = 1;   % [2,3] which classes belongs together

% Prototypes' labeling definition

prot_lbl = 1;               % = 1 / 2 / 3

% Metaparameters

MP.max_it = 020;   	% Maximum number of iterations (random search)
MP.fold = 5;     	% number of data partitions (cross validation)
MP.cost = 2;        % Takes into account also the dicitionary size
MP.lambda = 2.0;    % Jpbc = Ds + lambda * Err

%% CHOOSE FIXED HYPERPARAMETERS 

HP_ksomgd_lin.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_lin.ep = 200;          % max number of epochs
HP_ksomgd_lin.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_lin.init = 02;         % neurons' initialization
HP_ksomgd_lin.dist = 02;    	 % type of distance
HP_ksomgd_lin.learn = 02;   	 % type of learning step
HP_ksomgd_lin.No = 0.7;     	 % initial learning step
HP_ksomgd_lin.Nt = 0.01;         % final learning step
HP_ksomgd_lin.Nn = 01;           % number of neighbors
HP_ksomgd_lin.neig = 03;         % type of neighbor function
HP_ksomgd_lin.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_lin.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_lin.Von = 0;           % disable video
HP_ksomgd_lin.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_lin.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_lin.Ktype = 1;         % Type of Kernel
HP_ksomgd_lin.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_lin.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_lin.theta = 0;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_lin.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomgd_gau.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_gau.ep = 200;          % max number of epochs
HP_ksomgd_gau.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_gau.init = 02;         % neurons' initialization
HP_ksomgd_gau.dist = 02;    	 % type of distance
HP_ksomgd_gau.learn = 02;   	 % type of learning step
HP_ksomgd_gau.No = 0.7;     	 % initial learning step
HP_ksomgd_gau.Nt = 0.01;         % final learning step
HP_ksomgd_gau.Nn = 01;           % number of neighbors
HP_ksomgd_gau.neig = 03;         % type of neighbor function
HP_ksomgd_gau.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_gau.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_gau.Von = 0;           % disable video
HP_ksomgd_gau.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_gau.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_gau.Ktype = 2;         % Type of Kernel
HP_ksomgd_gau.sigma = 0.5;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_gau.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_gau.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_gau.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomgd_pol.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_pol.ep = 200;          % max number of epochs
HP_ksomgd_pol.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_pol.init = 02;         % neurons' initialization
HP_ksomgd_pol.dist = 02;    	 % type of distance
HP_ksomgd_pol.learn = 02;   	 % type of learning step
HP_ksomgd_pol.No = 0.7;     	 % initial learning step
HP_ksomgd_pol.Nt = 0.01;         % final learning step
HP_ksomgd_pol.Nn = 01;           % number of neighbors
HP_ksomgd_pol.neig = 03;         % type of neighbor function
HP_ksomgd_pol.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_pol.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_pol.Von = 0;           % disable video
HP_ksomgd_pol.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_pol.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_pol.Ktype = 3;         % Type of Kernel
HP_ksomgd_pol.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_pol.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_pol.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_pol.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomgd_exp.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_exp.ep = 200;          % max number of epochs
HP_ksomgd_exp.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_exp.init = 02;         % neurons' initialization
HP_ksomgd_exp.dist = 02;    	 % type of distance
HP_ksomgd_exp.learn = 02;   	 % type of learning step
HP_ksomgd_exp.No = 0.7;     	 % initial learning step
HP_ksomgd_exp.Nt = 0.01;         % final learning step
HP_ksomgd_exp.Nn = 01;           % number of neighbors
HP_ksomgd_exp.neig = 03;         % type of neighbor function
HP_ksomgd_exp.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_exp.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_exp.Von = 0;           % disable video
HP_ksomgd_exp.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_exp.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_exp.Ktype = 4;         % Type of Kernel
HP_ksomgd_exp.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_exp.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_exp.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_exp.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomgd_cau.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_cau.ep = 200;          % max number of epochs
HP_ksomgd_cau.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_cau.init = 02;         % neurons' initialization
HP_ksomgd_cau.dist = 02;    	 % type of distance
HP_ksomgd_cau.learn = 02;   	 % type of learning step
HP_ksomgd_cau.No = 0.7;     	 % initial learning step
HP_ksomgd_cau.Nt = 0.01;         % final learning step
HP_ksomgd_cau.Nn = 01;           % number of neighbors
HP_ksomgd_cau.neig = 03;         % type of neighbor function
HP_ksomgd_cau.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_cau.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_cau.Von = 0;           % disable video
HP_ksomgd_cau.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_cau.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_cau.Ktype = 5;         % Type of Kernel
HP_ksomgd_cau.sigma = 0.5;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_cau.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_cau.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_cau.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomgd_log.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_log.ep = 200;          % max number of epochs
HP_ksomgd_log.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_log.init = 02;         % neurons' initialization
HP_ksomgd_log.dist = 02;    	 % type of distance
HP_ksomgd_log.learn = 02;   	 % type of learning step
HP_ksomgd_log.No = 0.7;     	 % initial learning step
HP_ksomgd_log.Nt = 0.01;         % final learning step
HP_ksomgd_log.Nn = 01;           % number of neighbors
HP_ksomgd_log.neig = 03;         % type of neighbor function
HP_ksomgd_log.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_log.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_log.Von = 0;           % disable video
HP_ksomgd_log.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_log.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_log.Ktype = 6;         % Type of Kernel
HP_ksomgd_log.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_log.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_log.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_log.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomgd_sig.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_sig.ep = 200;          % max number of epochs
HP_ksomgd_sig.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_sig.init = 02;         % neurons' initialization
HP_ksomgd_sig.dist = 02;    	 % type of distance
HP_ksomgd_sig.learn = 02;   	 % type of learning step
HP_ksomgd_sig.No = 0.7;     	 % initial learning step
HP_ksomgd_sig.Nt = 0.01;         % final learning step
HP_ksomgd_sig.Nn = 01;           % number of neighbors
HP_ksomgd_sig.neig = 03;         % type of neighbor function
HP_ksomgd_sig.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_sig.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_sig.Von = 0;           % disable video
HP_ksomgd_sig.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_sig.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_sig.Ktype = 7;         % Type of Kernel
HP_ksomgd_sig.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_sig.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_sig.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_sig.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomgd_kmo.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd_kmo.ep = 200;          % max number of epochs
HP_ksomgd_kmo.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd_kmo.init = 02;         % neurons' initialization
HP_ksomgd_kmo.dist = 02;    	 % type of distance
HP_ksomgd_kmo.learn = 02;   	 % type of learning step
HP_ksomgd_kmo.No = 0.7;     	 % initial learning step
HP_ksomgd_kmo.Nt = 0.01;         % final learning step
HP_ksomgd_kmo.Nn = 01;           % number of neighbors
HP_ksomgd_kmo.neig = 03;         % type of neighbor function
HP_ksomgd_kmo.Vo = 0.8;          % initial neighbor constant
HP_ksomgd_kmo.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd_kmo.Von = 0;           % disable video
HP_ksomgd_kmo.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd_kmo.knn_type = 2;      % Type of knn aproximation
HP_ksomgd_kmo.Ktype = 8;         % Type of Kernel
HP_ksomgd_kmo.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomgd_kmo.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomgd_kmo.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomgd_kmo.gamma = 2;       	 % polynomial order (poly 2 or 3)

%% HYPERPARAMETERS - FOR OPTIMIZATION

if(~strcmp(OPT.hpo,'none'))

HPgs_ksomgd_lin = HP_ksomgd_lin;
HPgs_ksomgd_lin.theta = [0,2.^linspace(-4,3,8)];

HPgs_ksomgd_gau = HP_ksomgd_gau;
HPgs_ksomgd_gau.sigma = 2.^linspace(-6,5,12);

HPgs_ksomgd_pol = HP_ksomgd_pol;
HPgs_ksomgd_pol.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
HPgs_ksomgd_pol.alpha = 2.^linspace(-8,2,11); % 1;
HPgs_ksomgd_pol.theta = [0,2.^linspace(-4,3,8)]; % 0;

HPgs_ksomgd_exp = HP_ksomgd_exp;
HPgs_ksomgd_exp.sigma = 2.^linspace(-8,6,15);

HPgs_ksomgd_cau = HP_ksomgd_cau;
HPgs_ksomgd_cau.sigma = 2.^linspace(-8,6,15);

HPgs_ksomgd_log = HP_ksomgd_log;
HPgs_ksomgd_log.sigma = 2.^linspace(-8,6,15);
HPgs_ksomgd_log.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];

HPgs_ksomgd_sig = HP_ksomgd_sig;
HPgs_ksomgd_sig.alpha = 2.^linspace(-8,2,11);
HPgs_ksomgd_sig.theta = [0.001, 0.005, 0.01, 0.05, 0.1];

HPgs_ksomgd_kmo = HP_ksomgd_kmo;
HPgs_ksomgd_kmo.sigma = 2.^linspace(-8,6,15);
HPgs_ksomgd_kmo.gamma = 2.^linspace(-8,6,15);

end

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);         % See pairplot of attributes

[Nc,~] = size(DATA.output);         % Get number of classes

%% ACCUMULATORS AND HANDLERS

data_acc = cell(OPT.Nr,1);         	% Acc of labels and data division
NAMES = {'Linear','Gaussian',...    % Acc of names for plots
         'Polynomial', 'Exponential',...
         'Cauchy', 'Log',...
         'Sigmoid', 'Kmod'};    
nstats_all_tr = cell(length(NAMES),1);
nstats_all_ts = cell(length(NAMES),1);

ksomgd_lin_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_lin_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_lin_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_lin_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_lin_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomgd_gau_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_gau_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_gau_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_gau_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_gau_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomgd_pol_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_pol_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_pol_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_pol_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_pol_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomgd_exp_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_exp_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_exp_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_exp_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_exp_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomgd_cau_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_cau_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_cau_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_cau_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_cau_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomgd_log_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_log_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_log_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_log_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_log_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomgd_sig_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_sig_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_sig_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_sig_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_sig_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

ksomgd_kmo_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-GD
ksomgd_kmo_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_kmo_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_kmo_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomgd_kmo_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

%% FILE NAME

OPT.filename = strcat(DATA.name,'_prob2_',OPT.prob2,'_ksomgd','_hpo_',OPT.hpo,...
                      '_norm',int2str(OPT.norm),'_1nn');

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp(r);
display(datestr(now));

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
    HP_ksomgd_lin = random_search_cv(DATAtr,HPgs_ksomgd_lin,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomgd_gau = random_search_cv(DATAtr,HPgs_ksomgd_gau,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomgd_pol = random_search_cv(DATAtr,HPgs_ksomgd_pol,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomgd_exp = random_search_cv(DATAtr,HPgs_ksomgd_exp,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomgd_cau = random_search_cv(DATAtr,HPgs_ksomgd_cau,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomgd_log = random_search_cv(DATAtr,HPgs_ksomgd_log,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomgd_sig = random_search_cv(DATAtr,HPgs_ksomgd_sig,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomgd_kmo = random_search_cv(DATAtr,HPgs_ksomgd_kmo,...
                                     @ksom_gd_train,@ksom_gd_classify,MP);
end

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

ksomgd_lin_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_lin);
ksomgd_gau_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_gau);
ksomgd_pol_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_pol);
ksomgd_exp_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_exp);
ksomgd_cau_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_cau);
ksomgd_log_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_log);
ksomgd_sig_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_sig);
ksomgd_kmo_par_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd_kmo);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

ksomgd_lin_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_lin_par_acc{r});
ksomgd_lin_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_lin_out_tr_acc{r});
ksomgd_lin_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_lin_par_acc{r});
ksomgd_lin_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_lin_out_ts_acc{r});

ksomgd_gau_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_gau_par_acc{r});
ksomgd_gau_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_gau_out_tr_acc{r});
ksomgd_gau_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_gau_par_acc{r});
ksomgd_gau_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_gau_out_ts_acc{r});

ksomgd_pol_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_pol_par_acc{r});
ksomgd_pol_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_pol_out_tr_acc{r});
ksomgd_pol_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_pol_par_acc{r});
ksomgd_pol_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_pol_out_ts_acc{r});

ksomgd_exp_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_exp_par_acc{r});
ksomgd_exp_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_exp_out_tr_acc{r});
ksomgd_exp_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_exp_par_acc{r});
ksomgd_exp_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_exp_out_ts_acc{r});

ksomgd_cau_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_cau_par_acc{r});
ksomgd_cau_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_cau_out_tr_acc{r});
ksomgd_cau_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_cau_par_acc{r});
ksomgd_cau_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_cau_out_ts_acc{r});

ksomgd_log_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_log_par_acc{r});
ksomgd_log_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_log_out_tr_acc{r});
ksomgd_log_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_log_par_acc{r});
ksomgd_log_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_log_out_ts_acc{r});

ksomgd_sig_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_sig_par_acc{r});
ksomgd_sig_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_sig_out_tr_acc{r});
ksomgd_sig_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_sig_par_acc{r});
ksomgd_sig_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_sig_out_ts_acc{r});

ksomgd_kmo_out_tr_acc{r} = ksom_gd_classify(DATAtr,ksomgd_kmo_par_acc{r});
ksomgd_kmo_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_kmo_out_tr_acc{r});
ksomgd_kmo_out_ts_acc{r} = ksom_gd_classify(DATAts,ksomgd_kmo_par_acc{r});
ksomgd_kmo_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_kmo_out_ts_acc{r});

end

%% RESULTS / STATISTICS

% Statistics for n turns 

ksomgd_lin_nstats_tr = class_stats_nturns(ksomgd_lin_stats_tr_acc);
ksomgd_lin_nstats_ts = class_stats_nturns(ksomgd_lin_stats_ts_acc);

ksomgd_gau_nstats_tr = class_stats_nturns(ksomgd_gau_stats_tr_acc);
ksomgd_gau_nstats_ts = class_stats_nturns(ksomgd_gau_stats_ts_acc);

ksomgd_pol_nstats_tr = class_stats_nturns(ksomgd_pol_stats_tr_acc);
ksomgd_pol_nstats_ts = class_stats_nturns(ksomgd_pol_stats_ts_acc);

ksomgd_exp_nstats_tr = class_stats_nturns(ksomgd_exp_stats_tr_acc);
ksomgd_exp_nstats_ts = class_stats_nturns(ksomgd_exp_stats_ts_acc);

ksomgd_cau_nstats_tr = class_stats_nturns(ksomgd_cau_stats_tr_acc);
ksomgd_cau_nstats_ts = class_stats_nturns(ksomgd_cau_stats_ts_acc);

ksomgd_log_nstats_tr = class_stats_nturns(ksomgd_log_stats_tr_acc);
ksomgd_log_nstats_ts = class_stats_nturns(ksomgd_log_stats_ts_acc);

ksomgd_sig_nstats_tr = class_stats_nturns(ksomgd_sig_stats_tr_acc);
ksomgd_sig_nstats_ts = class_stats_nturns(ksomgd_sig_stats_ts_acc);

ksomgd_kmo_nstats_tr = class_stats_nturns(ksomgd_kmo_stats_tr_acc);
ksomgd_kmo_nstats_ts = class_stats_nturns(ksomgd_kmo_stats_ts_acc);

% Get all Statistics in one Cell

nstats_all_tr{1,1} = ksomgd_lin_nstats_tr;
nstats_all_tr{2,1} = ksomgd_gau_nstats_tr;
nstats_all_tr{3,1} = ksomgd_pol_nstats_tr;
nstats_all_tr{4,1} = ksomgd_exp_nstats_tr;
nstats_all_tr{5,1} = ksomgd_cau_nstats_tr;
nstats_all_tr{6,1} = ksomgd_log_nstats_tr;
nstats_all_tr{7,1} = ksomgd_sig_nstats_tr;
nstats_all_tr{8,1} = ksomgd_kmo_nstats_tr;

nstats_all_ts{1,1} = ksomgd_lin_nstats_ts;
nstats_all_ts{2,1} = ksomgd_gau_nstats_ts;
nstats_all_ts{3,1} = ksomgd_pol_nstats_ts;
nstats_all_ts{4,1} = ksomgd_exp_nstats_ts;
nstats_all_ts{5,1} = ksomgd_cau_nstats_ts;
nstats_all_ts{6,1} = ksomgd_log_nstats_ts;
nstats_all_ts{7,1} = ksomgd_sig_nstats_ts;
nstats_all_ts{8,1} = ksomgd_kmo_nstats_ts;

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all_tr,NAMES);

class_stats_ncomp(nstats_all_ts,NAMES);

%% SAVE DATA

if(OPT.savefile)
    variables.nstats_all_tr = nstats_all_tr;
    variables.nstats_all_ts = nstats_all_ts;
    variables.ksomgd_lin_par_acc = ksomgd_lin_par_acc;
    variables.ksomgd_gau_par_acc = ksomgd_gau_par_acc;
    variables.ksomgd_pol_par_acc = ksomgd_pol_par_acc;
    variables.ksomgd_exp_par_acc = ksomgd_exp_par_acc;
    variables.ksomgd_cau_par_acc = ksomgd_cau_par_acc;
    variables.ksomgd_log_par_acc = ksomgd_log_par_acc;
    variables.ksomgd_sig_par_acc = ksomgd_sig_par_acc;
    variables.ksomgd_kmo_par_acc = ksomgd_kmo_par_acc;
    save(OPT.filename,'variables');
    clear variables;
end

%% END