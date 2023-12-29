%% Machine Learning ToolBox

% KSOM Unit Test
% Author: David Nascimento Coelho
% Last Update: 2023/12/26

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

OPT.savefile = 1;   % decides if file will be saved

OPT.calculate_bin = 0;  % [0 or 1] decides to calculate binary statistics
OPT.class_1_vect = 1;   % [2,3] which classes belongs together

% Prototypes' labeling definition

prot_lbl = 1;               % = 1 (MV) / 2 (AD) / 3 (MD)

% Metaparameters

MP.max_it = 100;   	% Maximum number of iterations (random search)
MP.fold = 5;     	% number of data partitions (cross validation)
MP.cost = 2;        % Takes into account also the dicitionary size
MP.lambda = 2.0;    % Jpbc = Ds + lambda * Err

%% CHOOSE FIXED HYPERPARAMETERS

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
HP_ksomef.Ktype = 2;         % Type of Kernel
HP_ksomef.sigma = 0.5;   	 % Variance (gaussian, log, cauchy kernel)

%% CHOOSE HYPERPARAMETERS - FOR OPTIMIZATION

if(~strcmp(OPT.hpo,'none'))
    HP_ksomef_gs = HP_ksomef;
else
    % ToDo - Define search space of parameters
end

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);         % See pairplot of attributes

[Nc,~] = size(DATA.output);         % Get number of classes

%% ACCUMULATORS AND HANDLERS

NAMES = {'train','test'};               % Names for plots

data_acc = cell(OPT.Nr,1);              % Acc of labels and data division

nstats_all = cell(length(NAMES),1);     % 

ksomef_par_acc = cell(OPT.Nr,1);        % Acc Parameters of KSOM-EF
ksomef_out_tr_acc = cell(OPT.Nr,1);     % Acc of training data output
ksomef_out_ts_acc = cell(OPT.Nr,1);     % Acc of test data output
ksomef_stats_tr_acc = cell(OPT.Nr,1);   % Acc of training statistics
ksomef_stats_ts_acc = cell(OPT.Nr,1);   % Acc of test statistics

%% FILE NAME

OPT.filename = strcat(DATA.name,'_prob2_',int2str(OPT.prob2),'_ksomef',...
                      '_hpo_',OPT.hpo,'_norm_',int2str(OPT.norm), ...
                      '_nn_',int2str(HP_ksomef.K));

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
    HP_ksomef = random_search_cv(DATAtr,HP_ksomef_gs,...
                                 @ksom_ef_train,@ksom_ef_classify,MP);
end

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

ksomef_par_acc{r} = ksom_ef_train(DATAtr,HP_ksomef);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

ksomef_out_tr_acc{r} = ksom_ef_classify(DATAtr,ksomef_par_acc{r});
ksomef_stats_tr_acc{r} = class_stats_1turn(DATAtr,ksomef_out_tr_acc{r});

ksomef_out_ts_acc{r} = ksom_ef_classify(DATAts,ksomef_par_acc{r});
ksomef_stats_ts_acc{r} = class_stats_1turn(DATAts,ksomef_out_ts_acc{r});

end

%% RESULTS / STATISTICS

% Statistics for n turns (multiclass)

ksomef_nstats_tr = class_stats_nturns(ksomef_stats_tr_acc);
ksomef_nstats_ts = class_stats_nturns(ksomef_stats_ts_acc);

% Statistics for n turns (binary)

if(OPT.calculate_bin == 1)

end

% Get all Statistics in one Cell

nstats_all{1,1} = ksomef_nstats_tr;
nstats_all{2,1} = ksomef_nstats_ts;

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all,NAMES);

%% SAVE DATA

if(OPT.savefile)
    save(OPT.filename);
end

%% END