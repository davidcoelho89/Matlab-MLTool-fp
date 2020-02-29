%% Machine Learning ToolBox

% WSOM Journal 2018 Tests 1
% Author: David Nascimento Coelho
% Last Update: 2018/04/17

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 07;                % Which problem will be solved 
                              % Which data set will be used
OPT.prob2 = 01;               % When it needs an specification of data set
OPT.norm = 3;                 % Normalization definition
OPT.lbl = 0;                  % Data labeling definition
OPT.Nr = 50;                  % Number of repetitions of each algorithm
OPT.hold = 01;                % Hold out method
OPT.ptrn = 0.7;               % Percentage of samples for training
OPT.file = 'wsom2018_f5.mat'; % file where all the variables will be saved  

% Prototypes' labeling definition

prot_lbl = 3;               % = 1 / 2 / 3

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HIPERPARAMETERS - DEFAULT

SOM2p_acc = cell(OPT.Nr,1);	 % Init of Acc Hyperparameters of SOM-2D
PAR_SOM2d.lbl = prot_lbl;	 % Neurons' labeling function
PAR_SOM2d.ep = 200;          % max number of epochs
PAR_SOM2d.k = [5 4];         % number of neurons (prototypes)
PAR_SOM2d.init = 02;         % neurons' initialization
PAR_SOM2d.dist = 02;         % type of distance
PAR_SOM2d.learn = 02;        % type of learning step
PAR_SOM2d.No = 0.7;          % initial learning step
PAR_SOM2d.Nt = 0.01;         % final learnin step
PAR_SOM2d.Nn = 01;      	 % number of neighbors
PAR_SOM2d.neig = 03;         % type of neighborhood function
PAR_SOM2d.Vo = 0.8;          % initial neighborhood constant
PAR_SOM2d.Vt = 0.3;          % final neighborhood constant
PAR_SOM2d.Von = 0;           % disable video

KSOM1p_acc = cell(OPT.Nr,1);     % Init of Acc Hyperparameters of KSOM-PS
PAR_ksom_ps1.lbl = prot_lbl;     % Neurons' labeling function
PAR_ksom_ps1.ep = 200;           % max number of epochs
PAR_ksom_ps1.k = [5 4];          % number of neurons (prototypes)
PAR_ksom_ps1.init = 02;          % neurons' initialization
PAR_ksom_ps1.dist = 02;          % type of distance
PAR_ksom_ps1.learn = 02;         % type of learning step
PAR_ksom_ps1.No = 0.7;           % initial learning step
PAR_ksom_ps1.Nt = 0.01;          % final learnin step
PAR_ksom_ps1.Nn = 01;            % number of neighbors
PAR_ksom_ps1.neig = 03;          % type of neighborhood function
PAR_ksom_ps1.Vo = 0.8;           % initial neighborhood constant
PAR_ksom_ps1.Vt = 0.3;           % final neighborhood constant
PAR_ksom_ps1.Kt = 1;             % Type of Kernel
PAR_ksom_ps1.sig2 = 0.5;         % Variance (gaussian, log, cauchy kernel)
PAR_ksom_ps1.Von = 0;            % disable video
PAR_ksom_ps1.s = 2;              % prototype selection type
PAR_ksom_ps1.M = 50;             % samples used to estimate kernel matrix

KSOM2p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KSOM-GD1
PAR_ksom_gd.lbl = prot_lbl;     % Neurons' labeling function
PAR_ksom_gd.ep = 200;           % max number of epochs
PAR_ksom_gd.k = [5 4];          % number of neurons (prototypes)
PAR_ksom_gd.init = 02;          % neurons' initialization
PAR_ksom_gd.dist = 02;          % type of distance
PAR_ksom_gd.learn = 02;         % type of learning step
PAR_ksom_gd.No = 0.7;           % initial learning step
PAR_ksom_gd.Nt = 0.01;          % final learning step
PAR_ksom_gd.Nn = 01;            % number of neighbors
PAR_ksom_gd.neig = 03;          % type of neighbor function
PAR_ksom_gd.Vo = 0.8;           % initial neighbor constant
PAR_ksom_gd.Vt = 0.3;           % final neighbor constant
PAR_ksom_gd.Kt = 3;             % Type of Kernel
PAR_ksom_gd.sig2 = 0.5;         % Variance (gaussian, log, cauchy kernel)
PAR_ksom_gd.Von = 0;            % disable video

KSOM3p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KSOM-EF1
PAR_ksom_ef.lbl = prot_lbl;     % Neurons' labeling function
PAR_ksom_ef.ep = 200;           % max number of epochs
PAR_ksom_ef.k = [5 4];          % number of neurons (prototypes)
PAR_ksom_ef.init = 02;          % neurons' initialization
PAR_ksom_ef.dist = 02;          % type of distance
PAR_ksom_ef.learn = 02;         % type of learning step
PAR_ksom_ef.No = 0.7;           % initial learning step
PAR_ksom_ef.Nt = 0.01;          % final learning step
PAR_ksom_ef.Nn = 01;            % number of neighbors
PAR_ksom_ef.neig = 03;          % type of neighbor function
PAR_ksom_ef.Vo = 0.8;           % initial neighbor constant
PAR_ksom_ef.Vt = 0.3;           % final neighbor constant
PAR_ksom_ef.Kt = 3;             % Type of Kernel
PAR_ksom_ef.sig2 = 0.5;         % Variance (gaussian, log, cauchy kernel)
PAR_ksom_ef.Von = 0;            % disable video

KSOM4p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KSOM-PS
PAR_ksom_ps2.lbl = prot_lbl;     % Neurons' labeling function
PAR_ksom_ps2.ep = 200;           % max number of epochs
PAR_ksom_ps2.k = [5 4];          % number of neurons (prototypes)
PAR_ksom_ps2.init = 02;          % neurons' initialization
PAR_ksom_ps2.dist = 02;          % type of distance
PAR_ksom_ps2.learn = 02;         % type of learning step
PAR_ksom_ps2.No = 0.7;           % initial learning step
PAR_ksom_ps2.Nt = 0.01;          % final learnin step
PAR_ksom_ps2.Nn = 01;            % number of neighbors
PAR_ksom_ps2.neig = 03;          % type of neighborhood function
PAR_ksom_ps2.Vo = 0.8;           % initial neighborhood constant
PAR_ksom_ps2.Vt = 0.3;           % final neighborhood constant
PAR_ksom_ps2.Kt = 1;             % Type of Kernel
PAR_ksom_ps2.sig2 = 0.5;         % Variance (gaussian, log, cauchy kernel)
PAR_ksom_ps2.Von = 0;            % disable video
PAR_ksom_ps2.s = 3;              % prototype selection type
PAR_ksom_ps2.v = 0.01;           % accuracy parameter (level of sparsity)

KSOM5p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KSOM-PS
PAR_ksom_ps3.lbl = prot_lbl;     % Neurons' labeling function
PAR_ksom_ps3.ep = 200;           % max number of epochs
PAR_ksom_ps3.k = [5 4];          % number of neurons (prototypes)
PAR_ksom_ps3.init = 02;          % neurons' initialization
PAR_ksom_ps3.dist = 02;          % type of distance
PAR_ksom_ps3.learn = 02;         % type of learning step
PAR_ksom_ps3.No = 0.7;           % initial learning step
PAR_ksom_ps3.Nt = 0.01;          % final learnin step
PAR_ksom_ps3.Nn = 01;            % number of neighbors
PAR_ksom_ps3.neig = 03;          % type of neighborhood function
PAR_ksom_ps3.Vo = 0.8;           % initial neighborhood constant
PAR_ksom_ps3.Vt = 0.3;           % final neighborhood constant
PAR_ksom_ps3.Kt = 1;             % Type of Kernel
PAR_ksom_ps3.sig2 = 0.5;         % Variance (gaussian, log, cauchy kernel)
PAR_ksom_ps3.Von = 0;            % disable video
PAR_ksom_ps3.s = 1;              % prototype selection type
PAR_ksom_ps3.M = 50;             % samples used to estimate kernel matrix

%% CLASSIFIERS' RESULTS INIT

hold_acc = cell(OPT.Nr,1);          % Acc of labels and data division

som2_out_tr = cell(OPT.Nr,1);       % Acc of training data output
som2_out_ts = cell(OPT.Nr,1);       % Acc of test data output
som2_Mconf_sum = zeros(Nc,Nc);      % Aux var for mean confusion matrix calc

ksomps1_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ksomps1_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ksomps1_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

ksomgd_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ksomgd_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ksomgd_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

ksomef_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ksomef_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ksomef_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

ksomps2_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ksomps2_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ksomps2_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

ksomps3_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ksomps3_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ksomps3_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr,
    
% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

[DATAho] = hold_out(DATA,OPT);

hold_acc{r} = DATAho;
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;    

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

PAR_SOM2d = som2d_train(DATAtr);

PAR_ksom_ps1 = ksom_ps_train(DATAtr,PAR_ksom_ps1);

PAR_ksom_gd = ksom_gd_train(DATAtr,PAR_ksom_gd);

PAR_ksom_ef = ksom_ef_train(DATAtr,PAR_ksom_ef);

PAR_ksom_ps2 = ksom_ps_train(DATAtr,PAR_ksom_ps2);

PAR_ksom_ps3 = ksom_ps_train(DATAtr,PAR_ksom_ps3);

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

% KSOM-PS - With Nystrom Method

[OUTtr] = ksom_ps_classify(DATAtr,PAR_ksom_ps1);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomps1_out_tr{r,1} = OUTtr;                         % training set results

[OUTts] = ksom_ps_classify(DATAts,PAR_ksom_ps1);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomps1_out_ts{r,1} = OUTts;                         % test set results

KSOM1p_acc{r} = PAR_ksom_ps1;                       % hold parameters
ksomps1_Mconf_sum = ksomps1_Mconf_sum + OUTts.Mconf;  % hold confusion matrix

% KSOM-GD

[OUTtr] = ksom_gd_classify(DATAtr,PAR_ksom_gd);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomgd_out_tr{r,1} = OUTtr;                        % training set results

[OUTts] = ksom_gd_classify(DATAts,PAR_ksom_gd);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomgd_out_ts{r,1} = OUTts;                        % test set results

KSOM2p_acc{r} = PAR_ksom_gd;                       % hold parameters
ksomgd_Mconf_sum = ksomgd_Mconf_sum + OUTts.Mconf;	% hold confusion matrix

% KSOM-EF

[OUTtr] = ksom_ef_classify(DATAtr,PAR_ksom_ef);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomef_out_tr{r,1} = OUTtr;                        % training set results

[OUTts] = ksom_ef_classify(DATAts,PAR_ksom_ef);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomef_out_ts{r,1} = OUTts;                        % test set results

KSOM3p_acc{r} = PAR_ksom_ef;                       % hold parameters
ksomef_Mconf_sum = ksomef_Mconf_sum + OUTts.Mconf;  % hold confusion matrix

% KSOM-PS - ALD

[OUTtr] = ksom_ps_classify(DATAtr,PAR_ksom_ps2);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomps2_out_tr{r,1} = OUTtr;                         % training set results

[OUTts] = ksom_ps_classify(DATAts,PAR_ksom_ps2);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomps2_out_ts{r,1} = OUTts;                         % test set results

KSOM4p_acc{r} = PAR_ksom_ps2;                       % hold parameters
ksomps2_Mconf_sum = ksomps2_Mconf_sum + OUTts.Mconf;  % hold confusion matrix

% KSOM-PS - Random

[OUTtr] = ksom_ps_classify(DATAtr,PAR_ksom_ps3);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomps3_out_tr{r,1} = OUTtr;                         % training set results

[OUTts] = ksom_ps_classify(DATAts,PAR_ksom_ps3);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomps3_out_ts{r,1} = OUTts;                         % test set results

KSOM5p_acc{r} = PAR_ksom_ps3;                       % hold parameters
ksomps3_Mconf_sum = ksomps3_Mconf_sum + OUTts.Mconf;  % hold confusion matrix

end

%% STATISTICS

% Mean Confusion Matrix

som2_Mconf_sum = som2_Mconf_sum / OPT.Nr;
ksomps1_Mconf_sum = ksomps1_Mconf_sum / OPT.Nr;
ksomgd_Mconf_sum = ksomgd_Mconf_sum / OPT.Nr;
ksomef_Mconf_sum = ksomef_Mconf_sum / OPT.Nr;
ksomps2_Mconf_sum = ksomps2_Mconf_sum / OPT.Nr;
ksomps3_Mconf_sum = ksomps3_Mconf_sum / OPT.Nr;

som2_Mconf_sum2 = [som2_Mconf_sum(1,1) sum(som2_Mconf_sum(1,2:end)) ; sum(som2_Mconf_sum(2:end,1)) sum(sum(som2_Mconf_sum(2:end,2:end)))];
ksomps1_Mconf_sum2 = [ksomps1_Mconf_sum(1,1) sum(ksomps1_Mconf_sum(1,2:end)) ; sum(ksomps1_Mconf_sum(2:end,1)) sum(sum(ksomps1_Mconf_sum(2:end,2:end)))];
ksomgd1_Mconf_sum2 = [ksomgd_Mconf_sum(1,1) sum(ksomgd_Mconf_sum(1,2:end)) ; sum(ksomgd_Mconf_sum(2:end,1)) sum(sum(ksomgd_Mconf_sum(2:end,2:end)))];
ksomef1_Mconf_sum2 = [ksomef_Mconf_sum(1,1) sum(ksomef_Mconf_sum(1,2:end)) ; sum(ksomef_Mconf_sum(2:end,1)) sum(sum(ksomef_Mconf_sum(2:end,2:end)))];
ksomps2_Mconf_sum2 = [ksomps2_Mconf_sum(1,1) sum(ksomps2_Mconf_sum(1,2:end)) ; sum(ksomps2_Mconf_sum(2:end,1)) sum(sum(ksomps2_Mconf_sum(2:end,2:end)))];
ksomps3_Mconf_sum3 = [ksomps3_Mconf_sum(1,1) sum(ksomps3_Mconf_sum(1,2:end)) ; sum(ksomps3_Mconf_sum(2:end,1)) sum(sum(ksomps3_Mconf_sum(2:end,2:end)))];

%% GRAPHICS

% Init labels' cells and Init boxplot matrix

labels = {};

Mat_boxplot1 = []; % Train Multiclass
Mat_boxplot2 = []; % Train Binary
Mat_boxplot3 = []; % Test Multiclass
Mat_boxplot4 = []; % Test Binary

% SOM-2D

% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'SOM 2D'};
% Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(som2_out_tr)];
% Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(som2_out_tr)];
% Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(som2_out_ts)];
% Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(som2_out_ts)];

% KSOM-GD

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-GD'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomgd_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomgd_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomgd_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomgd_out_ts)];

% KSOM-EF

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-EF'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomef_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomef_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomef_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomef_out_ts)];

% KSOM-PS - Nystrom

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-PS-N'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomps1_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomps1_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomps1_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomps1_out_ts)];

% KSOM-PS - ALD

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-PS-A'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomps2_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomps2_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomps2_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomps2_out_ts)];

% KSOM-PS - RANDOM

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KSOM-PS-R'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ksomps3_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ksomps3_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ksomps3_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ksomps3_out_ts)];

% Generate Boxplots

% BOXPLOT 1
figure; boxplot(Mat_boxplot1, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto média
max1 = max(Mat_boxplot4);       % Taxa máxima de acerto
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
media2 = mean(Mat_boxplot2);    % Taxa de acerto média
max2 = max(Mat_boxplot4);       % Taxa máxima de acerto
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
media3 = mean(Mat_boxplot3);    % Taxa de acerto média
max3 = max(Mat_boxplot4);       % Taxa máxima de acerto
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
media4 = mean(Mat_boxplot4);    % Taxa de acerto média
max4 = max(Mat_boxplot4);       % Taxa máxima de acerto
plot(media4,'*k')
hold off

%% Best and Worst Mconf

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