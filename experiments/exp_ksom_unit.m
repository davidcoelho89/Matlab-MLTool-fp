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

% Statistics for n turns

som_nstats_tr = class_stats_nturns(som_stats_tr_acc);
som_nstats_ts = class_stats_nturns(som_stats_ts_acc);

ksomgd_nstats_tr = class_stats_nturns(ksomgd_stats_tr_acc);
ksomgd_nstats_ts = class_stats_nturns(ksomgd_stats_ts_acc);

ksomef_nstats_tr = class_stats_nturns(ksomef_stats_tr_acc);
ksomef_nstats_ts = class_stats_nturns(ksomef_stats_ts_acc);

% Get all Statistics in one Cell

nstats_all_tr{1,1} = som_nstats_tr;
nstats_all_tr{2,1} = ksomgd_nstats_tr;
nstats_all_tr{3,1} = ksomef_nstats_tr;

nstats_all_ts{1,1} = som_nstats_ts;
nstats_all_ts{2,1} = ksomgd_nstats_ts;
nstats_all_ts{3,1} = ksomef_nstats_ts;

% Compare Training and Test Statistics

class_stats_ncomp(nstats_all_tr,NAMES);

class_stats_ncomp(nstats_all_ts,NAMES);

% som_Mconf_sum2 = [som_Mconf_sum(1,1) sum(som_Mconf_sum(1,2:end)) ; sum(som_Mconf_sum(2:end,1)) sum(sum(som_Mconf_sum(2:end,2:end)))];
% ksomgd_Mconf_sum2 = [ksomgd_Mconf_sum(1,1) sum(ksomgd_Mconf_sum(1,2:end)) ; sum(ksomgd_Mconf_sum(2:end,1)) sum(sum(ksomgd_Mconf_sum(2:end,2:end)))];
% ksomef_Mconf_sum2 = [ksomef_Mconf_sum(1,1) sum(ksomef_Mconf_sum(1,2:end)) ; sum(ksomef_Mconf_sum(2:end,1)) sum(sum(ksomef_Mconf_sum(2:end,2:end)))];

%% GRAPHICS - CONSTRUCT

% Init labels' cells and Init boxplot matrix

% labels = {};
% 
% Mat_boxplot1 = []; % Train Multiclass
% Mat_boxplot2 = []; % Train Binary
% Mat_boxplot3 = []; % Test Multiclass
% Mat_boxplot4 = []; % Test Binary
% 
% % SOM-2D
% 
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'SOM 2D'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(som_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(som_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(som_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(som_out_ts)];
% 
% % KSOM-GD
% 
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'KSOM-GD'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomgd_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomgd_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomgd_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomgd_out_ts)];
% 
% % KSOM-EF
% 
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'KSOM-EF'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomef_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomef_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomef_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomef_out_ts)];

%% GRAPHICS - DISPLAY

% % BOXPLOT 1
% figure; boxplot(Mat_boxplot1, 'label', labels);
% set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
% ylabel('Accuracy')              % label eixo y
% xlabel('Classifiers')           % label eixo x
% title('Classifiers Comparison') % Titulo
% axis ([0 n_labels+1 0 1.05])	% Eixos
% 
% hold on
% media1 = mean(Mat_boxplot1);    % Taxa de acerto média
% max1 = max(Mat_boxplot1);       % Taxa máxima de acerto
% plot(media1,'*k')
% hold off
% 
% % BOXPLOT 2
% figure; boxplot(Mat_boxplot2, 'label', labels);
% set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
% ylabel('Accuracy')              % label eixo y
% xlabel('Classifiers')           % label eixo x
% title('Classifiers Comparison') % Titulo
% axis ([0 n_labels+1 0 1.05])	% Eixos
% 
% hold on
% media2 = mean(Mat_boxplot2);    % Taxa de acerto média
% max2 = max(Mat_boxplot2);       % Taxa máxima de acerto
% plot(media2,'*k')
% hold off
% 
% % BOXPLOT 3
% figure; boxplot(Mat_boxplot3, 'label', labels);
% set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
% ylabel('Accuracy')              % label eixo y
% xlabel('Classifiers')           % label eixo x
% title('Classifiers Comparison') % Titulo
% axis ([0 n_labels+1 0 1.05])	% Eixos
% 
% hold on
% media3 = mean(Mat_boxplot3);    % Taxa de acerto média
% max3 = max(Mat_boxplot3);       % Taxa máxima de acerto
% plot(media3,'*k')
% hold off
% 
% % BOXPLOT 4
% figure; boxplot(Mat_boxplot4, 'label', labels);
% set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
% ylabel('Accuracy')              % label eixo y
% xlabel('Classifiers')           % label eixo x
% title('Classifiers Comparison') % Titulo
% axis ([0 n_labels+1 0 1.05])	% Eixos
% 
% hold on
% media4 = mean(Mat_boxplot4);    % Taxa de acerto média
% max4 = max(Mat_boxplot4);       % Taxa máxima de acerto
% plot(media4,'*k')
% hold off

%% BEST AND WORST MCONF

% results_to_test = som2_out_ts;
% 
% x1 = accuracy_bin(results_to_test);
% 
% [~,max_mconf] = max(x1);
% [~,min_mconf] = min(x1);
% 
% Mconf_max = results_to_test{max_mconf,1}.Mconf;
% Mconf_min = results_to_test{min_mconf,1}.Mconf;

%% SAVE DATA

if(OPT.savefile)
    save(OPT.filename);
end

%% END