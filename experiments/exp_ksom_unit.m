%% Machine Learning ToolBox

% KSOM Comparison Tests
% Author: David Nascimento Coelho
% Last Update: 2023/02/08

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 03;        % Number of repetitions of each algorithm
OPT.alg = 'ksom';   % Which Classifier will be used
OPT.prob = 06;      % Which problem will be solved / used
OPT.prob2 = 01;     % When it needs an specification of data set
OPT.norm = 0;       % Normalization definition
OPT.lbl = 1;        % Data labeling definition
OPT.hold = 02;      % Hold out method
OPT.ptrn = 0.7;     % Percentage of samples for training
OPT.hpo = 'none';   % 'grid' ; 'random' ; 'none'

OPT.savefile = 0;   % decides if file will be saved

OPT.calculate_bin = 0;  % decides to calculate binary statistics
OPT.class_1_vect = 1;   % [2,3] which classes belongs together

% Prototypes' labeling definition

prot_lbl = 1;               % = 1 / 2 / 3

% Metaparameters

MP.max_it = 100;   	% Maximum number of iterations (random search)
MP.fold = 5;     	% number of data partitions (cross validation)
MP.cost = 2;        % Takes into account also the dicitionary size
MP.lambda = 2.0;    % Jpbc = Ds + lambda * Err

%% CHOOSE FIXED HYPERPARAMETERS 

HP_som.lbl = prot_lbl;      % Neurons' labeling function
HP_som.ep = 200;            % max number of epochs
HP_som.k = [5 4];           % number of neurons (prototypes)
HP_som.init = 02;           % neurons' initialization
HP_som.dist = 02;           % type of distance
HP_som.learn = 02;          % type of learning step
HP_som.No = 0.7;            % initial learning step
HP_som.Nt = 0.01;           % final learnin step
HP_som.Nn = 01;             % number of neighbors
HP_som.neig = 03;           % type of neighborhood function
HP_som.Vo = 0.8;            % initial neighborhood constant
HP_som.Vt = 0.3;            % final neighborhood constant
HP_som.Von = 0;             % disable video
HP_som.K = 1;            	% Number of nearest neighbors (classify)
HP_som.knn_type = 2;        % Type of knn aproximation
HP_som.Ktype = 0;           % Non-kernelized Algorithm

HP_ksomgd.lbl = prot_lbl;    % Neurons' labeling function
HP_ksomgd.ep = 200;          % max number of epochs
HP_ksomgd.k = [5 4];         % number of neurons (prototypes)
HP_ksomgd.init = 02;         % neurons' initialization
HP_ksomgd.dist = 02;    	 % type of distance
HP_ksomgd.learn = 02;   	 % type of learning step
HP_ksomgd.No = 0.7;     	 % initial learning step
HP_ksomgd.Nt = 0.01;         % final learning step
HP_ksomgd.Nn = 01;           % number of neighbors
HP_ksomgd.neig = 03;         % type of neighbor function
HP_ksomgd.Vo = 0.8;          % initial neighbor constant
HP_ksomgd.Vt = 0.3;     	 % final neighbor constant
HP_ksomgd.Von = 0;           % disable video
HP_ksomgd.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomgd.knn_type = 2;      % Type of knn aproximation
HP_ksomgd.Ktype = 1;         % Type of Kernel
HP_ksomgd.sigma = 0.5;   	 % Variance (gaussian, log, cauchy kernel)

HP_ksomef.lbl = prot_lbl;	 % Neurons' labeling function
HP_ksomef.ep = 200;          % max number of epochs
HP_ksomef.k = [5 4];         % number of neurons (prototypes)
HP_ksomef.init = 02;         % neurons' initialization
HP_ksomef.dist = 02;         % type of distance
HP_ksomef.learn = 02;        % type of learning step
HP_ksomef.No = 0.7;          % initial learning step
HP_ksomef.Nt = 0.01;         % final learning step
HP_ksomef.Nn = 01;           % number of neighbors
HP_ksomef.neig = 03;         % type of neighbor function
HP_ksomef.Vo = 0.8;          % initial neighbor constant
HP_ksomef.Vt = 0.3;          % final neighbor constant
HP_ksomef.Von = 0;           % disable video
HP_ksomef.K = 1;         	 % Number of nearest neighbors (classify)
HP_ksomef.knn_type = 2; 	 % Type of knn aproximation
HP_ksomef.Ktype = 1;         % Type of Kernel
HP_ksomef.sigma = 0.5;   	 % Variance (gaussian, log, cauchy kernel)

%% HYPERPARAMETERS - FOR OPTIMIZATION

if(~strcmp(OPT.hpo,'none'))

HP_som_gs = HP_som;

HP_ksomgd_gs = HP_ksomgd;

HP_ksomef_gs = HP_ksomef;

% ToDo - Define grid of parameters

end

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);         % See pairplot of attributes

[Nc,~] = size(DATA.output);         % Get number of classes

%% ACCUMULATORS AND HANDLERS

data_acc = cell(OPT.Nr,1);            % Acc of labels and data division
NAMES = {'som','ksomgd','ksomef'};    % Acc of names for plots

nstats_all_tr = cell(length(NAMES),1);
nstats_all_ts = cell(length(NAMES),1);

SOMp_acc = cell(OPT.Nr,1);	          % Acc Parameters of SOM
som_out_tr = cell(OPT.Nr,1);          % Acc of training data output
som_out_ts = cell(OPT.Nr,1);          % Acc of test data output
som_stats_tr_acc = cell(OPT.Nr,1);    % Acc of training statistics
som_stats_ts_acc = cell(OPT.Nr,1);    % Acc of test statistics
som_Mconf_sum = zeros(Nc,Nc);         % Aux var for mean conf mat calc

KSOMGDp_acc = cell(OPT.Nr,1);         % Acc Parameters of KSOM-GD
ksomgd_out_tr = cell(OPT.Nr,1);	      % Acc of training data output
ksomgd_out_ts = cell(OPT.Nr,1);	      % Acc of test data output
ksomgd_stats_tr_acc = cell(OPT.Nr,1); % Acc of training statistics
ksomgd_stats_ts_acc = cell(OPT.Nr,1); % Acc of test statistics
ksomgd_Mconf_sum = zeros(Nc,Nc);      % Aux var for mean conf mat calc

KSOMEFp_acc = cell(OPT.Nr,1);           % Acc Parameters of KSOM-EF
ksomef_out_tr = cell(OPT.Nr,1);	        % Acc of training data output
ksomef_out_ts = cell(OPT.Nr,1);	        % Acc of test data output
ksomef_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics
ksomef_Mconf_sum = zeros(Nc,Nc);        % Aux var for mean conf mat calc

%% FILE NAME

OPT.filename = strcat(DATA.name,'_ksom_',...
                      'protLbl',int2str(prot_lbl));

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
    HP_som = random_search_cv(DATAtr,HP_som_gs,...
                              @som_train,@som_classify,MP);
    HP_ksomgd = random_search_cv(DATAtr,HP_ksomgd_gs,...
                                 @ksom_gd_train,@ksom_gd_classify,MP);
    HP_ksomef = random_search_cv(DATAtr,HP_ksomef_gs,...
                                 @ksom_ef_train,@ksom_ef_classify,MP);
end

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

SOMp_acc{r} = som_train(DATAtr,HP_som);

KSOMGDp_acc{r} = ksom_gd_train(DATAtr,HP_ksomgd);

KSOMEFp_acc{r} = ksom_ef_train(DATAtr,HP_ksomef);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

% SOM

som_out_tr{r} = som_classify(DATAtr,SOMp_acc{r});
som_stats_tr_acc{r} = class_stats_1turn(DATAtr,som_out_tr{r});

som_out_ts{r} = som_classify(DATAts,SOMp_acc{r});
som_stats_ts_acc{r} = class_stats_1turn(DATAts,som_out_ts{r});

% KSOM-GD

ksomgd_out_tr{r} = ksom_gd_classify(DATAtr,KSOMGDp_acc{r});
ksomgd_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomgd_out_tr{r});

ksomgd_out_ts{r} = ksom_gd_classify(DATAts,KSOMGDp_acc{r});
ksomgd_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomgd_out_ts{r});

% KSOM-EF

ksomef_out_tr{r} = ksom_ef_classify(DATAtr,KSOMEFp_acc{r});
ksomef_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_out_tr{r});

ksomef_out_ts{r} = ksom_ef_classify(DATAts,KSOMEFp_acc{r});
ksomef_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_out_ts{r});

end

%% RESULTS / STATISTICS

% ToDo - Calculate "Binary" Statistics for comparison
% - Define which classes will be in class "1"
% - Calculate statistics from already calculated Mconf

% Statistics for n turns (multiclass)

som_nstats_tr = class_stats_nturns(som_stats_tr_acc);
som_nstats_ts = class_stats_nturns(som_stats_ts_acc);

ksomgd_nstats_tr = class_stats_nturns(ksomgd_stats_tr_acc);
ksomgd_nstats_ts = class_stats_nturns(ksomgd_stats_ts_acc);

ksomef_nstats_tr = class_stats_nturns(ksomef_stats_tr_acc);
ksomef_nstats_ts = class_stats_nturns(ksomef_stats_ts_acc);

% Statistics for n turns (binary)

som_stats_tr_bin_acc = calculate_binary_stats(som_stats_tr_acc,OPT.class_1_vect);
som_stats_ts_bin_acc = calculate_binary_stats(som_stats_ts_acc,OPT.class_1_vect);
som_nstats_tr_bin = class_stats_nturns(som_stats_tr_bin_acc);
som_nstats_ts_bin = class_stats_nturns(som_stats_ts_bin_acc);

ksomgd_stats_tr_bin_acc = calculate_binary_stats(ksomgd_stats_tr_acc,OPT.class_1_vect);
ksomgd_stats_ts_bin_acc = calculate_binary_stats(ksomgd_stats_ts_acc,OPT.class_1_vect);
ksomgd_nstats_tr_bin = class_stats_nturns(ksomgd_stats_tr_bin_acc);
ksomgd_nstats_ts_bin = class_stats_nturns(ksomgd_stats_ts_bin_acc);

ksomef_stats_tr_bin_acc = calculate_binary_stats(ksomef_stats_tr_acc,OPT.class_1_vect);
ksomef_stats_ts_bin_acc = calculate_binary_stats(ksomef_stats_ts_acc,OPT.class_1_vect);
ksomef_nstats_tr_bin = class_stats_nturns(ksomef_stats_tr_bin_acc);
ksomef_nstats_ts_bin = class_stats_nturns(ksomef_stats_ts_bin_acc);

% Get all Statistics in one Cell

nstats_all_tr{1,1} = som_nstats_tr;
nstats_all_tr{2,1} = ksomgd_nstats_tr;
nstats_all_tr{3,1} = ksomef_nstats_tr;

nstats_all_ts{1,1} = som_nstats_ts;
nstats_all_ts{2,1} = ksomgd_nstats_ts;
nstats_all_ts{3,1} = ksomef_nstats_ts;

nstats_all_tr_bin{1,1} = som_nstats_tr_bin;
nstats_all_tr_bin{2,1} = ksomgd_nstats_tr_bin;
nstats_all_tr_bin{3,1} = ksomef_nstats_tr_bin;

nstats_all_ts_bin{1,1} = som_nstats_ts_bin;
nstats_all_ts_bin{2,1} = ksomgd_nstats_ts_bin;
nstats_all_ts_bin{3,1} = ksomef_nstats_ts_bin;

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all_tr,NAMES);
class_stats_ncomp(nstats_all_ts,NAMES);
class_stats_ncomp(nstats_all_tr_bin,NAMES);
class_stats_ncomp(nstats_all_ts_bin,NAMES);

%% SAVE DATA

if(OPT.savefile)
    save(OPT.filename);
end

%% END