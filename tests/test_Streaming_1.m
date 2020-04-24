%% Machine Learning ToolBox

% Online and Sequential Algorithms
% Author: David Nascimento Coelho
% Last Update: 2020/02/23

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 25;              % Which problem will be solved / used
OPT.prob2 = 30;             % More details about a specific data set
OPT.norm = 0;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 50;              	% Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% CHOOSE HYPERPARAMETERS

% Kernel Functions: 1 lin / 2 gauss / 3 poly / 5 cauchy / 6 log / 7 sigm /

HP.Dm = 2;          % Design Method
HP.Ss = 1;          % Sparsification strategy
HP.v1 = 0.5;        % Sparseness parameter 1 
HP.v2 = 0.9;        % Sparseness parameter 2
HP.Ps = 1;          % Prunning strategy
HP.min_score = -10; % Score that leads the sample to be pruned
HP.Us = 1;          % Update strategy
HP.eta = 0.01;      % Update rate
HP.max_prot = Inf;  % Max number of prototypes
HP.Von = 1;         % Enable / disable video 
HP.K = 1;           % Number of nearest neighbors (classify)
HP.Ktype = 2;       % Kernel Type ( 2 Gauss / 3 poly / 5 cauc / 7 sigm)
HP.sig2n = 0.001;   % Kernel Regularization parameter
HP.sigma = 7;    	% Kernel width (gaussian)
HP.gamma = 2;       % polynomial order (poly 2 or 3)
HP.alpha = 1;       % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 1;       % Dot product add cte (lin 0 / poly 1 / sigm 0.1)

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

%% VISUALIZE DATA

% For SEA data

DATA1 = DATA;
DATA1.input = DATA.input(:,1:5000);
DATA1.output = DATA.output(:,1:5000);
 
figure; plot_data_pairplot(DATA1);             % visualize data

% For Iris

% figure; plot_data_pairplot(DATA)

%% ACCUMULATORS

NAMES = {'train','test'};           % Hold names for plots
DATA_acc = cell(OPT.Nr,1);       	% Hold Data
PAR_acc = cell(OPT.Nr,1);         	% Hold Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Hold Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Hold Statistics of test data
nSTATS_all = cell(2,1);             % Hold General statistics

accuracy_vector = zeros(1,N);       % Hold Acc / (Acc + Err)
no_of_correct = zeros(1,N);         % Hold # of correctly classified x
no_of_errors = zeros(1,N);          % Hold # of misclassified x
predict_vector = zeros(2,N);        % Hold true and predicted labels
no_of_samples = zeros(Nc,N);        % Hold number of samples per class

figure; VID = struct('cdata',cell(1,N),'colormap', cell(1,N));

%% SEQUENTIAL TESTS AND STATISTICS

% Shuffle Data
% I = randperm(size(DATA.input,2));
% DATA.input = DATA.input(:,I);
% DATA.output = DATA.output(:,I);
% DATA.lbl = DATA.lbl(:,I);

% Get statistics from data (For Video Function)
DATAn.Xmax = max(DATA.input,[],2);  % max value
DATAn.Xmin = min(DATA.input,[],2);  % min value
DATAn.Xmed = mean(DATA.input,2);    % mean value
DATAn.Xdp = std(DATA.input,[],2);   % std value

% Add first element to dictionary
DATAn.input = DATA.input(:,1);      % First element input
DATAn.output = DATA.output(:,1);    % First element output
[~,max_y] = max(DATAn.output);      % Get sample's class   
no_of_samples(max_y,1) = 1;         % Update number of samples per class
PAR = k2nn_train(DATAn,HP);         % Add element

% Update Video Function
if (HP.Von),
    VID(1) = prototypes_frame(PAR.Cx,DATAn);
end

for n = 2:N,
    
    % Display number of samples already seen (for debug)
    
    if(mod(n,1000) == 0),
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data (and hold its class)
    
    DATAn.input = DATA.input(:,n);
    DATAn.output = DATA.output(:,n);
    
    [~,max_y] = max(DATAn.output);
    predict_vector(1,n) = max_y;
    
    for c = 1:Nc,
        if (c == max_y),
            no_of_samples(c,n) = no_of_samples(c,n-1) + 1;
        else
            no_of_samples(c,n) = no_of_samples(c,n-1);
        end
    end
    
    % Test (classify arriving data with current model)
    
    OUTn = k2nn_classify(DATAn,PAR);

    [~,max_yh] = max(OUTn.y_h);
    predict_vector(2,n) = max_yh;
    
    % Statistics
    
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

% Data and Prototypes
figure;
hold on 
plot(DATA.input(1,:),DATA.input(2,:),'r.');
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
hold off

% Number of hits x number of errors
figure;
hold on
plot(x,no_of_errors,'r-');
plot(x,no_of_correct,'b-');
hold off

% Percentage of Correct Classified
figure;
plot(x,accuracy_vector,'r-');

% Number of samples per class
figure;
colors = lines(Nc);
hold on
for c = 1:Nc,
    plot(x,no_of_samples(c,:),'Color',colors(c,:));
end
hold off

%% SAVE FILE



%% END