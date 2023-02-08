%% Machine Learning ToolBox

% KSOM Comparison Tests
% Author: David Nascimento Coelho
% Last Update: 2023/02/08

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.Nr = 50;        % Number of repetitions of each algorithm
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

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);         % See pairplot of attributes

[Nc,~] = size(DATA.output);         % Get number of classes

%% HYPERPARAMETERS - DEFAULT

SOMp_acc = cell(OPT.Nr,1);	 % Init of Acc Hyperparameters of SOM-2D
PAR_SOM.lbl = prot_lbl;	     % Neurons' labeling function
PAR_SOM.ep = 200;            % max number of epochs
PAR_SOM.k = [5 4];           % number of neurons (prototypes)
PAR_SOM.init = 02;           % neurons' initialization
PAR_SOM.dist = 02;           % type of distance
PAR_SOM.learn = 02;          % type of learning step
PAR_SOM.No = 0.7;            % initial learning step
PAR_SOM.Nt = 0.01;           % final learnin step
PAR_SOM.Nn = 01;      	     % number of neighbors
PAR_SOM.neig = 03;           % type of neighborhood function
PAR_SOM.Vo = 0.8;            % initial neighborhood constant
PAR_SOM.Vt = 0.3;            % final neighborhood constant
PAR_SOM.Von = 0;             % disable video
PAR_SOM.K = 1;               % nearest neighbor scheme
PAR_SOM.Ktype = 0;           % Non-kernelized Algorithm

KSOM1p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-GD1
PAR_ksom_gd.lbl = prot_lbl;  % Neurons' labeling function
PAR_ksom_gd.ep = 200;  	     % max number of epochs
PAR_ksom_gd.k = [5 4];   	 % number of neurons (prototypes)
PAR_ksom_gd.init = 02;   	 % neurons' initialization
PAR_ksom_gd.dist = 02;    	 % type of distance
PAR_ksom_gd.learn = 02;   	 % type of learning step
PAR_ksom_gd.No = 0.7;     	 % initial learning step
PAR_ksom_gd.Nt = 0.01;   	 % final learning step
PAR_ksom_gd.Nn = 01;     	 % number of neighbors
PAR_ksom_gd.neig = 03;   	 % type of neighbor function
PAR_ksom_gd.Vo = 0.8;    	 % initial neighbor constant
PAR_ksom_gd.Vt = 0.3;     	 % final neighbor constant
PAR_ksom_gd.Kt = 1;       	 % Type of Kernel
PAR_ksom_gd.sig2 = 0.5;   	 % Variance (gaussian, log, cauchy kernel)
PAR_ksom_gd.Von = 0;         % disable video

KSOM4p_acc = cell(OPT.Nr,1); % Init of Acc Hyperparameters of KSOM-EF1
PAR_ksom_ef.lbl = prot_lbl;  % Neurons' labeling function
PAR_ksom_ef.ep = 200;        % max number of epochs
PAR_ksom_ef.k = [5 4];       % number of neurons (prototypes)
PAR_ksom_ef.init = 02;       % neurons' initialization
PAR_ksom_ef.dist = 02;       % type of distance
PAR_ksom_ef.learn = 02;      % type of learning step
PAR_ksom_ef.No = 0.7;        % initial learning step
PAR_ksom_ef.Nt = 0.01;       % final learning step
PAR_ksom_ef.Nn = 01;         % number of neighbors
PAR_ksom_ef.neig = 03;       % type of neighbor function
PAR_ksom_ef.Vo = 0.8;        % initial neighbor constant
PAR_ksom_ef.Vt = 0.3;        % final neighbor constant
PAR_ksom_ef.Kt = 1;          % Type of Kernel
PAR_ksom_ef.sig2 = 0.5;      % Variance (gaussian, log, cauchy kernel)
PAR_ksom_ef.Von = 0;         % disable video

%% CLASSIFIERS' RESULTS INIT

hold_acc = cell(OPT.Nr,1);          % Acc of labels and data division

som_out_tr = cell(OPT.Nr,1);        % Acc of training data output
som_out_ts = cell(OPT.Nr,1);        % Acc of test data output
som_Mconf_sum = zeros(Nc,Nc);       % Aux var for mean confusion matrix calc

ksomgd_out_tr = cell(OPT.Nr,1);	    % Acc of training data output
ksomgd_out_ts = cell(OPT.Nr,1);	    % Acc of test data output
ksomgd_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

ksomef_out_tr = cell(OPT.Nr,1);	    % Acc of training data output
ksomef_out_ts = cell(OPT.Nr,1);	    % Acc of test data output
ksomef_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

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

hold_acc{r} = DATAho;
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



% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

PAR_SOM = som_train(DATAtr,PAR_SOM);

PAR_ksom_gd = ksom_gd_train(DATAtr,PAR_ksom_gd);

PAR_ksom_ef = ksom_ef_train(DATAtr,PAR_ksom_ef);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

% SOM

[OUTtr] = som_classify(DATAtr,PAR_SOM);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
som_out_tr{r,1} = OUTtr;                       % training set results

[OUTts] = som2d_classify(DATAts,PAR_SOM);
OUTts.nf = normal_or_fail(OUTts.Mconf);
som_out_ts{r,1} = OUTts;                       % test set results

SOMp_acc{r} = PAR_SOM;                       % hold parameters
som_Mconf_sum = som_Mconf_sum + OUTts.Mconf;  % hold confusion matrix

% KSOM-GD

[OUTtr] = ksom_gd_classify(DATAtr,PAR_ksom_gd);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomgd_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_gd_classify(DATAts,PAR_ksom_gd);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomgd_out_ts{r,1} = OUTts;                            % test set results

KSOM1p_acc{r} = PAR_ksom_gd;                           % hold parameters
ksomgd_Mconf_sum = ksomgd_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

% KSOM-EF

[OUTtr] = ksom_ef_classify(DATAtr,PAR_ksom_ef);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
ksomef_out_tr{r,1} = OUTtr;                            % training set results

[OUTts] = ksom_ef_classify(DATAts,PAR_ksom_ef);
OUTts.nf = normal_or_fail(OUTts.Mconf);
ksomef_out_ts{r,1} = OUTts;                            % test set results

KSOM4p_acc{r} = PAR_ksom_ef;                           % hold parameters
ksomef_Mconf_sum = ksomef_Mconf_sum + OUTts.Mconf;    % hold confusion matrix

end

%% STATISTICS

% Mean Confusion Matrix

som_Mconf_sum = som_Mconf_sum / OPT.Nr;
ksomgd_Mconf_sum = ksomgd_Mconf_sum / OPT.Nr;
ksomef_Mconf_sum = ksomef_Mconf_sum / OPT.Nr;

som_Mconf_sum2 = [som_Mconf_sum(1,1) sum(som_Mconf_sum(1,2:end)) ; sum(som_Mconf_sum(2:end,1)) sum(sum(som_Mconf_sum(2:end,2:end)))];
ksomgd_Mconf_sum2 = [ksomgd_Mconf_sum(1,1) sum(ksomgd_Mconf_sum(1,2:end)) ; sum(ksomgd_Mconf_sum(2:end,1)) sum(sum(ksomgd_Mconf_sum(2:end,2:end)))];
ksomef_Mconf_sum2 = [ksomef_Mconf_sum(1,1) sum(ksomef_Mconf_sum(1,2:end)) ; sum(ksomef_Mconf_sum(2:end,1)) sum(sum(ksomef_Mconf_sum(2:end,2:end)))];

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
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(som_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(som_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(som_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(som_out_ts)];

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

%% GRAPHICS - DISPLAY

% BOXPLOT 1
figure; boxplot(Mat_boxplot1, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto média
max1 = max(Mat_boxplot1);       % Taxa máxima de acerto
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
max2 = max(Mat_boxplot2);       % Taxa máxima de acerto
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
max3 = max(Mat_boxplot3);       % Taxa máxima de acerto
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