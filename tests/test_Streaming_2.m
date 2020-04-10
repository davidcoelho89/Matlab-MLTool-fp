%% Machine Learning ToolBox

% Online and Sequential Algorithms
% Author: David Nascimento Coelho
% Last Update: 2020/04/08

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 33;              % Which problem will be solved / used
OPT.prob2 = 30;             % More details about a specific data set
OPT.norm = 0;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 50;              	% Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% HYPERPARAMETERS - DEFAULT

% Kernel Functions: 1 lin / 2 gauss / 3 poly / 5 cauchy / 6 log / 7 sigm /

HP.Dm = 2;          % Design Method
HP.Ss = 1;          % Sparsification strategy
HP.v1 = 0.9;        % Sparseness parameter 1 
HP.v2 = 0.9;        % Sparseness parameter 2
HP.Ps = 1;          % Prunning strategy
HP.min_score = -10; % Score that leads the sample to be pruned
HP.Us = 1;          % Update strategy
HP.eta = 0.01;      % Update rate
HP.max_prot = Inf;  % Max number of prototypes
HP.Von = 0;         % Enable / disable video 
HP.K = 1;           % Number of nearest neighbors (classify)
HP.Ktype = 2;       % Kernel Type ( 2 Gauss / 3 poly / 5 cauc / 7 sigm)
HP.sig2n = 0.001;   % Kernel Regularization parameter
HP.sigma = 7;    	% Kernel width (gaussian)
HP.gamma = 2;       % polynomial order (poly 2 or 3)
HP.alpha = 1;       % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 1;       % Dot product add cte (lin 0 / poly 1 / sigm 0.1)

%% HYPERPARAMETERS - GRID FOR TRAINING AND TESTING



%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

% Set data for the cross validation step

if (N < 5000),
    Nhpo = floor(0.2 * N);
else
    Nhpo = 1000;
end

DATAhpo.input = DATA.input(:,1:Nhpo);
DATAhpo.output = DATA.output(:,1:Nhpo);

% Set remaining data for test-than-train step

DATAttt.input = DATA.input(:,Nhpo+1:end);
DATAttt.output = DATA.output(:,Nhpo+1:end);

%% VISUALIZE DATA

figure; plot_data_pairplot(DATAttt)

%% ACCUMULATORS

accuracy_vector = zeros(1,N);       % Hold Acc / (Acc + Err)
no_of_correct = zeros(1,N);         % Hold # of correctly classified x
no_of_errors = zeros(1,N);          % Hold # of misclassified x
predict_vector = zeros(2,N);        % Hold true and predicted labels
no_of_samples = zeros(Nc,N);        % Hold number of samples per class

figure; VID = struct('cdata',cell(1,N),'colormap', cell(1,N));

%% CROSS VALIDATION FOR HYPERPARAMETERS OPTIMIZATION




%% ADD FIRST ELEMENT TO DICTIONARY

% Get first element of dataset
DATAn.input = DATA.input(:,1);      % first element input
DATAn.output = DATA.output(:,1);    % first element output
DATAn.Xmax = max(DATA.input,[],2);  % max value
DATAn.Xmin = min(DATA.input,[],2);  % min value
DATAn.Xmed = mean(DATA.input,2);    % mean value
DATAn.Xdp = std(DATA.input,[],2);   % std value

% Add element to dictionary
[~,max_y] = max(DATAn.output);      
no_of_samples(max_y,1) = 1;
PAR = k2nn_train(DATAn,HP);

% Update Video Function
if (HP.Von),
    VID(1) = prototypes_frame(PAR.Cx,DATAn);
end

%% PRESEQUENTIAL (TEST-THAN-TRAIN)

for n = Nhpo+1:N,
    
end

%% PLOTS



%% SAVE FILE



%% END