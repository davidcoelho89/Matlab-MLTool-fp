%% Machine Learning ToolBox

% KSOM-EF Comparison Tests
% Author: David Nascimento Coelho
% Last Update: 2023/03/05

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 05;        % Number of repetitions of each algorithm
OPT.alg = 'ksomef'; % Which Classifier will be used
OPT.prob = 07;      % Which problem will be solved / used
OPT.prob2 = 01;     % When it needs an specification of data set
OPT.norm = 3;       % Normalization definition
OPT.lbl = 1;        % Data labeling definition
OPT.hold = 02;      % Hold out method
OPT.ptrn = 0.7;     % Percentage of samples for training
OPT.hpo = 'none'; % 'grid' ; 'random' ; 'none'

OPT.savefile = 0;   % decides if file will be saved

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

HP_ksomef_lin.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_lin.ep = 200;          % max number of epochs
HP_ksomef_lin.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_lin.init = 02;         % neurons' initialization
HP_ksomef_lin.dist = 02;    	 % type of distance
HP_ksomef_lin.learn = 02;   	 % type of learning step
HP_ksomef_lin.No = 0.7;     	 % initial learning step
HP_ksomef_lin.Nt = 0.01;         % final learning step
HP_ksomef_lin.Nn = 01;           % number of neighbors
HP_ksomef_lin.neig = 03;         % type of neighbor function
HP_ksomef_lin.Vo = 0.8;          % initial neighbor constant
HP_ksomef_lin.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_lin.Von = 0;           % disable video
HP_ksomef_lin.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_lin.knn_type = 2;      % Type of knn aproximation
HP_ksomef_lin.Ktype = 1;         % Type of Kernel
HP_ksomef_lin.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_lin.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_lin.theta = 0;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_lin.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomef_gau.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_gau.ep = 200;          % max number of epochs
HP_ksomef_gau.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_gau.init = 02;         % neurons' initialization
HP_ksomef_gau.dist = 02;    	 % type of distance
HP_ksomef_gau.learn = 02;   	 % type of learning step
HP_ksomef_gau.No = 0.7;     	 % initial learning step
HP_ksomef_gau.Nt = 0.01;         % final learning step
HP_ksomef_gau.Nn = 01;           % number of neighbors
HP_ksomef_gau.neig = 03;         % type of neighbor function
HP_ksomef_gau.Vo = 0.8;          % initial neighbor constant
HP_ksomef_gau.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_gau.Von = 0;           % disable video
HP_ksomef_gau.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_gau.knn_type = 2;      % Type of knn aproximation
HP_ksomef_gau.Ktype = 2;         % Type of Kernel
HP_ksomef_gau.sigma = 0.5;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_gau.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_gau.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_gau.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomef_pol.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_pol.ep = 200;          % max number of epochs
HP_ksomef_pol.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_pol.init = 02;         % neurons' initialization
HP_ksomef_pol.dist = 02;    	 % type of distance
HP_ksomef_pol.learn = 02;   	 % type of learning step
HP_ksomef_pol.No = 0.7;     	 % initial learning step
HP_ksomef_pol.Nt = 0.01;         % final learning step
HP_ksomef_pol.Nn = 01;           % number of neighbors
HP_ksomef_pol.neig = 03;         % type of neighbor function
HP_ksomef_pol.Vo = 0.8;          % initial neighbor constant
HP_ksomef_pol.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_pol.Von = 0;           % disable video
HP_ksomef_pol.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_pol.knn_type = 2;      % Type of knn aproximation
HP_ksomef_pol.Ktype = 3;         % Type of Kernel
HP_ksomef_pol.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_pol.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_pol.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_pol.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomef_exp.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_exp.ep = 200;          % max number of epochs
HP_ksomef_exp.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_exp.init = 02;         % neurons' initialization
HP_ksomef_exp.dist = 02;    	 % type of distance
HP_ksomef_exp.learn = 02;   	 % type of learning step
HP_ksomef_exp.No = 0.7;     	 % initial learning step
HP_ksomef_exp.Nt = 0.01;         % final learning step
HP_ksomef_exp.Nn = 01;           % number of neighbors
HP_ksomef_exp.neig = 03;         % type of neighbor function
HP_ksomef_exp.Vo = 0.8;          % initial neighbor constant
HP_ksomef_exp.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_exp.Von = 0;           % disable video
HP_ksomef_exp.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_exp.knn_type = 2;      % Type of knn aproximation
HP_ksomef_exp.Ktype = 4;         % Type of Kernel
HP_ksomef_exp.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_exp.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_exp.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_exp.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomef_cau.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_cau.ep = 200;          % max number of epochs
HP_ksomef_cau.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_cau.init = 02;         % neurons' initialization
HP_ksomef_cau.dist = 02;    	 % type of distance
HP_ksomef_cau.learn = 02;   	 % type of learning step
HP_ksomef_cau.No = 0.7;     	 % initial learning step
HP_ksomef_cau.Nt = 0.01;         % final learning step
HP_ksomef_cau.Nn = 01;           % number of neighbors
HP_ksomef_cau.neig = 03;         % type of neighbor function
HP_ksomef_cau.Vo = 0.8;          % initial neighbor constant
HP_ksomef_cau.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_cau.Von = 0;           % disable video
HP_ksomef_cau.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_cau.knn_type = 2;      % Type of knn aproximation
HP_ksomef_cau.Ktype = 5;         % Type of Kernel
HP_ksomef_cau.sigma = 0.5;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_cau.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_cau.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_cau.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomef_log.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_log.ep = 200;          % max number of epochs
HP_ksomef_log.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_log.init = 02;         % neurons' initialization
HP_ksomef_log.dist = 02;    	 % type of distance
HP_ksomef_log.learn = 02;   	 % type of learning step
HP_ksomef_log.No = 0.7;     	 % initial learning step
HP_ksomef_log.Nt = 0.01;         % final learning step
HP_ksomef_log.Nn = 01;           % number of neighbors
HP_ksomef_log.neig = 03;         % type of neighbor function
HP_ksomef_log.Vo = 0.8;          % initial neighbor constant
HP_ksomef_log.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_log.Von = 0;           % disable video
HP_ksomef_log.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_log.knn_type = 2;      % Type of knn aproximation
HP_ksomef_log.Ktype = 6;         % Type of Kernel
HP_ksomef_log.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_log.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_log.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_log.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomef_sig.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_sig.ep = 200;          % max number of epochs
HP_ksomef_sig.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_sig.init = 02;         % neurons' initialization
HP_ksomef_sig.dist = 02;    	 % type of distance
HP_ksomef_sig.learn = 02;   	 % type of learning step
HP_ksomef_sig.No = 0.7;     	 % initial learning step
HP_ksomef_sig.Nt = 0.01;         % final learning step
HP_ksomef_sig.Nn = 01;           % number of neighbors
HP_ksomef_sig.neig = 03;         % type of neighbor function
HP_ksomef_sig.Vo = 0.8;          % initial neighbor constant
HP_ksomef_sig.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_sig.Von = 0;           % disable video
HP_ksomef_sig.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_sig.knn_type = 2;      % Type of knn aproximation
HP_ksomef_sig.Ktype = 7;         % Type of Kernel
HP_ksomef_sig.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_sig.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_sig.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_sig.gamma = 2;       	 % polynomial order (poly 2 or 3)

HP_ksomef_kmo.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomef_kmo.ep = 200;          % max number of epochs
HP_ksomef_kmo.k = [5 4];         % number of neurons (prototypes)
HP_ksomef_kmo.init = 02;         % neurons' initialization
HP_ksomef_kmo.dist = 02;    	 % type of distance
HP_ksomef_kmo.learn = 02;   	 % type of learning step
HP_ksomef_kmo.No = 0.7;     	 % initial learning step
HP_ksomef_kmo.Nt = 0.01;         % final learning step
HP_ksomef_kmo.Nn = 01;           % number of neighbors
HP_ksomef_kmo.neig = 03;         % type of neighbor function
HP_ksomef_kmo.Vo = 0.8;          % initial neighbor constant
HP_ksomef_kmo.Vt = 0.3;     	 % final neighbor constant
HP_ksomef_kmo.Von = 0;           % disable video
HP_ksomef_kmo.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef_kmo.knn_type = 2;      % Type of knn aproximation
HP_ksomef_kmo.Ktype = 8;         % Type of Kernel
HP_ksomef_kmo.sigma = 2;     	 % Kernel width (gau, exp, cauchy, log, kmod)
HP_ksomef_kmo.alpha = 0.1;    	 % Dot product multiplier (poly 1 / sigm 0.1)
HP_ksomef_kmo.theta = 0.1;     	 % Dot product adding (poly 1 / sigm 0.1)
HP_ksomef_kmo.gamma = 2;       	 % polynomial order (poly 2 or 3)

%% HYPERPARAMETERS - FOR OPTIMIZATION

if(~strcmp(OPT.hpo,'none'))

HPgs_ksomef_lin = HP_ksomef_lin;
HPgs_ksomef_lin.theta = [0,2.^linspace(-4,3,8)];

HPgs_ksomef_gau = HP_ksomef_gau;
HPgs_ksomef_gau.sigma = 2.^linspace(-6,5,12);

HPgs_ksomef_pol = HP_ksomef_pol;
HPgs_ksomef_pol.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
HPgs_ksomef_pol.alpha = 2.^linspace(-8,2,11); % 1;
HPgs_ksomef_pol.theta = [0,2.^linspace(-4,3,8)]; % 0;

HPgs_ksomef_exp = HP_ksomef_exp;
HPgs_ksomef_exp.sigma = 2.^linspace(-8,6,15);

HPgs_ksomef_cau = HP_ksomef_cau;
HPgs_ksomef_cau.sigma = 2.^linspace(-8,6,15);

HPgs_ksomef_log = HP_ksomef_log;
HPgs_ksomef_log.sigma = 2.^linspace(-8,6,15);
HPgs_ksomef_log.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];

HPgs_ksomef_sig = HP_ksomef_sig;
HPgs_ksomef_sig.alpha = 2.^linspace(-8,2,11);
HPgs_ksomef_sig.theta = [0.001, 0.005, 0.01, 0.05, 0.1];

HPgs_ksomef_kmo = HP_ksomef_kmo;
HPgs_ksomef_kmo.sigma = 2.^linspace(-8,6,15);
HPgs_ksomef_kmo.gamma = 2.^linspace(-8,6,15);

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

OPT.filename = strcat(DATA.name,'_','ksomef','_hpo_',OPT.hpo,...
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

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all_tr,NAMES);

class_stats_ncomp(nstats_all_ts,NAMES);

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
    save(OPT.filename,'variables');
    clear variables;
end

%% END