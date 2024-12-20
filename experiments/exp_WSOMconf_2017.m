%% Machine Learning ToolBox

% WSOM Conference 2017 Tests
% Author: David Nascimento Coelho
% Last Update: 2019/01/22

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.Nr = 50;                  % Number of repetitions of each algorithm
OPT.prob = 07;                % Which problem will be solved / used
OPT.prob2 = 01;               % When it needs an specification of data set
OPT.norm = 3;                 % Normalization definition
OPT.lbl = 1;                  % Data labeling definition
OPT.hold = 01;                % Hold out method
OPT.ptrn = 0.7;               % Percentage of samples for training
OPT.file = 'wsom2017_f1.mat'; % file where all the variables will be saved  

% Prototypes' labeling definition

prot_lbl = 1;               % = 1 / 2 / 3

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HYPERPARAMETERS - DEFAULT

PAR_common.lbl = prot_lbl;    % Neurons' labeling function
PAR_common.ep = 200;          % max number of epochs
PAR_common.k = [5 4];         % number of neurons (prototypes)
PAR_common.init = 02;         % neurons' initialization
PAR_common.dist = 02;         % type of distance
PAR_common.learn = 02;        % type of learning step
PAR_common.No = 0.7;          % initial learning step
PAR_common.Nt = 0.01;         % final learnin step
PAR_common.Nn = 01;      	  % number of neighbors
PAR_common.neig = 03;         % type of neighborhood function
PAR_common.Vo = 0.8;          % initial neighborhood constant
PAR_common.Vt = 0.3;          % final neighborhood constant
PAR_common.Von = 0;           % disable video

SOM2p_acc = cell(OPT.Nr,1);	 % Init of Acc Hyperparameters of SOM-2D
PAR_SOM2d = PAR_common;      % Get common parameters
PAR_SOM2d.Kt = 0;            % Type of Kernel

KSOM1p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-GD1
PAR_ksom_gd1 = PAR_common;	 % Get common parameters
PAR_ksom_gd1.Kt = 1;       	 % Type of Kernel
PAR_ksom_gd1.sig2 = 0.5;   	 % Variance (gaussian, log, cauchy kernel)

KSOM2p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-GD2
PAR_ksom_gd2 = PAR_common;	 % Get common parameters
PAR_ksom_gd2.Kt = 2;       	 % Type of Kernel
PAR_ksom_gd2.sig2 = 0.5;   	 % Variance (gaussian, log, cauchy kernel)

KSOM3p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-GD3
PAR_ksom_gd3 = PAR_common;	 % Get common parameters
PAR_ksom_gd3.Kt = 3;         % Type of Kernel
PAR_ksom_gd3.sig2 = 0.5;     % Variance (gaussian, log, cauchy kernel)

KSOM4p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-EF1
PAR_ksom_ef1 = PAR_common;	 % Get common parameters
PAR_ksom_ef1.Kt = 1;         % Type of Kernel
PAR_ksom_ef1.sig2 = 0.5;     % Variance (gaussian, log, cauchy kernel)

KSOM5p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-EF2
PAR_ksom_ef2 = PAR_common;	 % Get common parameters
PAR_ksom_ef2.Kt = 2;         % Type of Kernel
PAR_ksom_ef2.sig2 = 0.5;     % Variance (gaussian, log, cauchy kernel)

KSOM6p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-EF3
PAR_ksom_ef3 = PAR_common;	 % Get common parameters
PAR_ksom_ef3.Kt = 3;         % Type of Kernel
PAR_ksom_ef3.sig2 = 0.5;     % Variance (gaussian, log, cauchy kernel)

%% CLASSIFIERS' RESULTS INIT

hold_acc = cell(OPT.Nr,1);          % Acc of labels and data division

som2_out_tr = cell(OPT.Nr,1);       % Acc of training data output
som2_out_ts = cell(OPT.Nr,1);       % Acc of test data output
som2_Mconf_sum = zeros(Nc,Nc);      % Aux var for mean confusion matrix calc

ksomps_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ksomps_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ksomps_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

ksomgd1_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ksomgd1_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ksomgd1_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

ksomgd2_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ksomgd2_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ksomgd2_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

ksomgd3_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ksomgd3_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ksomgd3_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

ksomef1_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ksomef1_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ksomef1_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

ksomef2_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ksomef2_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ksomef2_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

ksomef3_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ksomef3_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ksomef3_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

[DATAho] = hold_out(DATA,OPT);

hold_acc{r} = DATAho;
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%%



% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

PAR_SOM2d = som2d_train(DATAtr);

PAR_ksom_gd1 = ksom_gd_train(DATAtr,PAR_ksom_gd1);

PAR_ksom_gd2 = ksom_gd_train(DATAtr,PAR_ksom_gd2);

PAR_ksom_gd3 = ksom_gd_train(DATAtr,PAR_ksom_gd3);

PAR_ksom_ef1 = ksom_ef_train(DATAtr,PAR_ksom_ef1);

PAR_ksom_ef2 = ksom_ef_train(DATAtr,PAR_ksom_ef2);

PAR_ksom_ef3 = ksom_ef_train(DATAtr,PAR_ksom_ef3);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

% SOM 2D

[OUTtr] = som2d_classify(DATAtr,PAR_SOM2d);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
som2_out_tr{r,1} = OUTtr;                       % training set results

[OUTts] = som2d_classify(DATAts,PAR_SOM2d);
OUTts.nf = normal_or_fail(OUTts.Mconf);
som2_out_ts{r,1} = OUTts;                       % test set results

SOM2p_acc{r} = PAR_SOM2d;                       % hold parameters
som2_Mconf_sum = som2_Mconf_sum + OUTts.Mconf;  % hold confusion matrix

% KSOM-GD-1

[OUTtr] = ksom_gd_classify(DATAtr,PAR_ksom_gd1);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomgd1_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_gd_classify(DATAts,PAR_ksom_gd1);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomgd1_out_ts{r,1} = OUTts;                            % test set results

KSOM1p_acc{r} = PAR_ksom_gd1;                           % hold parameters
ksomgd1_Mconf_sum = ksomgd1_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

% KSOM-GD-2 (Cauchy)

[OUTtr] = ksom_gd_classify(DATAtr,PAR_ksom_gd2);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomgd2_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_gd_classify(DATAts,PAR_ksom_gd2);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomgd2_out_ts{r,1} = OUTts;                            % test set results

KSOM2p_acc{r} = PAR_ksom_gd2;                         	% hold parameters
ksomgd2_Mconf_sum = ksomgd2_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

% KSOM-GD-3 (Log)

[OUTtr] = ksom_gd_classify(DATAtr,PAR_ksom_gd3);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomgd3_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_gd_classify(DATAts,PAR_ksom_gd3);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomgd3_out_ts{r,1} = OUTts;                            % test set results

KSOM3p_acc{r} = PAR_ksom_gd3;                         	% hold parameters
ksomgd3_Mconf_sum = ksomgd3_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

% KSOM-EF-1 (Gaussian)

[OUTtr] = ksom_ef_classify(DATAtr,PAR_ksom_ef1);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomef1_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_ef_classify(DATAts,PAR_ksom_ef1);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomef1_out_ts{r,1} = OUTts;                            % test set results

KSOM4p_acc{r} = PAR_ksom_ef1;                           % hold parameters
ksomef1_Mconf_sum = ksomef1_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

% KSOM-EF-2 (Cauchy)

[OUTtr] = ksom_ef_classify(DATAtr,PAR_ksom_ef2);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomef2_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_ef_classify(DATAts,PAR_ksom_ef2);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomef2_out_ts{r,1} = OUTts;                            % test set results

KSOM5p_acc{r} = PAR_ksom_ef2;                           % hold parameters
ksomef2_Mconf_sum = ksomef2_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

% KSOM-EF-3 (Log)

[OUTtr] = ksom_ef_classify(DATAtr,PAR_ksom_ef3);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomef3_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_ef_classify(DATAts,PAR_ksom_ef3);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomef3_out_ts{r,1} = OUTts;                            % test set results

KSOM6p_acc{r} = PAR_ksom_ef3;                           % hold parameters
ksomef3_Mconf_sum = ksomef3_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

end

%% STATISTICS

% Mean Confusion Matrix

som2_Mconf_sum = som2_Mconf_sum / OPT.Nr;
ksomps_Mconf_sum = ksomps_Mconf_sum / OPT.Nr;
ksomgd1_Mconf_sum = ksomgd1_Mconf_sum / OPT.Nr;
ksomgd2_Mconf_sum = ksomgd2_Mconf_sum / OPT.Nr;
ksomgd3_Mconf_sum = ksomgd3_Mconf_sum / OPT.Nr;
ksomef1_Mconf_sum = ksomef1_Mconf_sum / OPT.Nr;
ksomef2_Mconf_sum = ksomef2_Mconf_sum / OPT.Nr;
ksomef3_Mconf_sum = ksomef3_Mconf_sum / OPT.Nr;

som2_Mconf_sum2 = [som2_Mconf_sum(1,1) sum(som2_Mconf_sum(1,2:end)) ; sum(som2_Mconf_sum(2:end,1)) sum(sum(som2_Mconf_sum(2:end,2:end)))];
ksomgd1_Mconf_sum2 = [ksomgd1_Mconf_sum(1,1) sum(ksomgd1_Mconf_sum(1,2:end)) ; sum(ksomgd1_Mconf_sum(2:end,1)) sum(sum(ksomgd1_Mconf_sum(2:end,2:end)))];
ksomgd2_Mconf_sum2 = [ksomgd2_Mconf_sum(1,1) sum(ksomgd2_Mconf_sum(1,2:end)) ; sum(ksomgd2_Mconf_sum(2:end,1)) sum(sum(ksomgd2_Mconf_sum(2:end,2:end)))];
ksomgd3_Mconf_sum2 = [ksomgd3_Mconf_sum(1,1) sum(ksomgd3_Mconf_sum(1,2:end)) ; sum(ksomgd3_Mconf_sum(2:end,1)) sum(sum(ksomgd3_Mconf_sum(2:end,2:end)))];
ksomef1_Mconf_sum2 = [ksomef1_Mconf_sum(1,1) sum(ksomef1_Mconf_sum(1,2:end)) ; sum(ksomef1_Mconf_sum(2:end,1)) sum(sum(ksomef1_Mconf_sum(2:end,2:end)))];
ksomef2_Mconf_sum2 = [ksomef2_Mconf_sum(1,1) sum(ksomef2_Mconf_sum(1,2:end)) ; sum(ksomef2_Mconf_sum(2:end,1)) sum(sum(ksomef2_Mconf_sum(2:end,2:end)))];
ksomef3_Mconf_sum2 = [ksomef3_Mconf_sum(1,1) sum(ksomef3_Mconf_sum(1,2:end)) ; sum(ksomef3_Mconf_sum(2:end,1)) sum(sum(ksomef3_Mconf_sum(2:end,2:end)))];

%% GRAPHICS - CONSTRUCT

% Init labels' cells and Init boxplot matrix

labels = {};

Mat_boxplot1 = []; % Train Multiclass
Mat_boxplot2 = []; % Train Binary
Mat_boxplot3 = []; % Test Multiclass
Mat_boxplot4 = []; % Test Binary

% SOM-2D

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'SOM 2D'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(som2_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(som2_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(som2_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(som2_out_ts)];

% KSOM-GD

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-GD-G'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomgd1_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomgd1_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomgd1_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomgd1_out_ts)];

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-GD-L'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomgd2_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomgd2_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomgd2_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomgd2_out_ts)];

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-GD-C'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomgd3_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomgd3_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomgd3_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomgd3_out_ts)];

% KSOM-EF

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-EF-G'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomef1_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomef1_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomef1_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomef1_out_ts)];

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-EF-L'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomef2_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomef2_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomef2_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomef2_out_ts)];

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-EF-C'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomef3_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomef3_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomef3_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomef3_out_ts)];

%% GRAPHICS - DISPLAY

% BOXPLOT 1
figure; boxplot(Mat_boxplot1, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto m�dia
max1 = max(Mat_boxplot1);       % Taxa m�xima de acerto
plot(media1,'*k')
hold off

% BOXPLOT 2
figure; boxplot(Mat_boxplot2, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media2 = mean(Mat_boxplot2);    % Taxa de acerto m�dia
max2 = max(Mat_boxplot2);       % Taxa m�xima de acerto
plot(media2,'*k')
hold off

% BOXPLOT 3
figure; boxplot(Mat_boxplot3, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media3 = mean(Mat_boxplot3);    % Taxa de acerto m�dia
max3 = max(Mat_boxplot3);       % Taxa m�xima de acerto
plot(media3,'*k')
hold off

% BOXPLOT 4
figure; boxplot(Mat_boxplot4, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media4 = mean(Mat_boxplot4);    % Taxa de acerto m�dia
max4 = max(Mat_boxplot4);       % Taxa m�xima de acerto
plot(media4,'*k')
hold off

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

save(OPT.file);

%% END