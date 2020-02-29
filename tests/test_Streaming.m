%% Machine Learning ToolBox

% Online and Sequential Algorithms
% Author: David Nascimento Coelho
% Last Update: 2020/02/23

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% OBSERVATIONS

% Contribuição Teórica: kernels
% Polinomial: d = 2 ou 3 ou fracionario. alpha = 1 e c = 1 (sem HP)
% (alpha * x' * y + c)
% Tg Hiperb: alpha = 1 e c = 1 -> 0.1 e 0.1 (sem HP)
% (alpha * x' * y + c)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 06;              % Which problem will be solved / used
OPT.prob2 = 30;             % More details about a specific data set
OPT.norm = 0;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 50;              	% Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% CHOOSE HYPERPARAMETERS

HP.Dm = 2;          % Design Method
HP.Ss = 1;          % Sparsification strategy
HP.v1 = 0.4;        % Sparseness parameter 1 
HP.v2 = 0.9;        % Sparseness parameter 2
HP.Ps = 1;          % Prunning strategy
HP.min_score = -10; % Score that leads the sample to be pruned
HP.Us = 1;          % Update strategy
HP.eta = 0.01;      % Update rate
HP.Von = 1;         % Enable / disable video 
HP.K = 1;           % Number of nearest neighbors (classify)
HP.Ktype = 2;       % Kernel Type ( 2 Gauss / 3 poly / 5 cauc / 7 sigm)
HP.sig2n = 0.001;   % Kernel Regularization parameter
HP.sigma = 2;    	% Kernel width (gaussian)
HP.alpha = 1;       % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 1;       % Dot product adding (poly 1 / sigm 0.1)
HP.order = 2;       % polynomial order

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[~,N] = size(DATA.input);        	% get number of attributes and samples

DATAn = struct('input',[],'output',[]);

%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

accuracy_vector = zeros(1,N);
no_of_correct = zeros(1,N);
no_of_errors = zeros(1,N);

VID = struct('cdata',cell(1,N),'colormap', cell(1,N));

%% SEQUENTIAL TESTS AND STATISTICS

% Shuffle Data
% I = randperm(size(DATA.input,2));
% DATA.input = DATA.input(:,I);
% DATA.output = DATA.output(:,I);
% DATA.lbl = DATA.lbl(:,I);

% Get max and min values of data
% And update "prototypes_frame" function!! (axis fixed)

% Get first element to dictionary
DATAn.input = DATA.input(:,1);
DATAn.output = DATA.output(:,1);
PAR = k2nn_train(DATAn,HP);

if (HP.Von),
    VID(1) = prototypes_frame(PAR.Cx,DATAn);
end

for n = 2:N,
    
    % Display number of samples already seen (for debug)
    if(mod(n,1000) == 0),
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data
    DATAn.input = DATA.input(:,n);
    DATAn.output = DATA.output(:,n);
    
    % Test (classify arring data with current model)
    OUTn = k2nn_classify(DATAn,PAR);
    
    % Statistics
    [~,max_y] = max(DATAn.output);
    [~,max_yh] = max(OUTn.y_h);
    if (max_y == max_yh),
        no_of_correct(n) = no_of_correct(n-1) + 1;
        no_of_errors(n) = no_of_errors(n-1);
    else
        no_of_correct(n) = no_of_correct(n-1);
        no_of_errors(n) = no_of_errors(n-1) + 1;
    end
    accuracy_vector(n) = no_of_correct(n) / ...
                        (no_of_correct(n) + no_of_errors(n));
    
    % Update score (for prunning method)
    PAR = k2nn_score_updt(DATAn,PAR,OUTn);
    
    % Train (with arriving data)
    PAR = k2nn_train(DATAn,PAR);
    
    % Video Function
    if (HP.Von),
        VID(n) = prototypes_frame(PAR.Cx,DATAn);
    end
    
end

%% PLOTS

x = 1:N;

figure;
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
hold on 
plot(DATA.input(1,:),DATA.input(2,:),'r.');
hold off

figure;
plot(x,no_of_errors,'r-');
hold on
plot(x,no_of_correct,'b-');
hold off

figure;
plot(x,accuracy_vector,'r-');

%% SAVE FILE



%% END