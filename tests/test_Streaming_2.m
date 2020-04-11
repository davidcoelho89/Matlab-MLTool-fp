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

%% CHOOSE ALGORITHM

class_name = 'k2nn';
class_train = @k2nn_train;
class_test = @k2nn_classify;

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

%% HYPERPARAMETERS - FOR GRID SEARCH

HP_gs = HP;
HP_gs.v1 = 2.^linspace(-10,10,21);

% Kernel Functions: 1 lin / 2 gauss / 3 poly / 5 cauchy / 6 log / 7 sigm /

% % linear 1
% K2NNcv.v1 = 2.^linspace(-10,10,21);
% % Gaussian 2
% K2NNcv.v1 = 2.^linspace(-4,3,8);
% K2NNcv.sigma = 2.^linspace(-10,9,20);
% % Polynomial 3
% K2NNcv.v1 = 2.^linspace(-13,6,20);
% K2NNcv.gamma = [2,3];
% % Cauchy 5
% K2NNcv.v1 = 2.^linspace(-4,3,8);
% K2NNcv.sigma = 2.^linspace(-10,9,20);
% % Log 6
% K2NNcv.v1 = -2.^linspace(10,2,9);
% K2NNcv.sigma = [0.001 0.01 0.1 1 2 5];
% % Sigm 7
% K2NNcv.v1 = 2.^linspace(-13,6,20);
% K2NNcv.alpha = 2.^linspace(-8,2,11);
% K2NNcv.theta = 2.^linspace(-8,2,11);

%% DATA LOADING AND PRE-PROCESSING

% Load Dataset and Adjust its Labels

DATA = data_class_loading(OPT);     % Load Data Set
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

% Set data for the cross validation step
% min (0.2 * N, 1000)

if (N < 5000),
    Nhpo = floor(0.2 * N);
else
    Nhpo = 1000;
end

DATAhpo.input = DATA.input(:,1:Nhpo);
DATAhpo.output = DATA.output(:,1:Nhpo);

% Set remaining data for test-than-train step

Nttt = N - Nhpo;

DATAttt.input = DATA.input(:,Nhpo+1:end);
DATAttt.output = DATA.output(:,Nhpo+1:end);

%% DATA NORMALIZATION

% Normalize hpo data
DATAhpo = normalize(DATAhpo,OPT);

% Normalize ttt data
DATAttt.Xmax = DATAhpo.Xmax;
DATAttt.Xmin = DATAhpo.Xmin;
DATAttt.Xmed = DATAhpo.Xmed;
DATAttt.Xdp = DATAhpo.Xdp;
DATAttt = normalize(DATAttt,OPT);

%% DATA VISUALIZATION

figure; plot_data_pairplot(DATAttt);

%% ACCUMULATORS

accuracy_vector = zeros(1,Nttt);       % Hold Acc / (Acc + Err)
no_of_correct = zeros(1,Nttt);         % Hold # of correctly classified x
no_of_errors = zeros(1,Nttt);          % Hold # of misclassified x
predict_vector = zeros(2,Nttt);        % Hold true and predicted labels
no_of_samples = zeros(Nc,Nttt);        % Hold number of samples per class

figure; VID = struct('cdata',cell(1,Nttt),'colormap', cell(1,Nttt));

%% CROSS VALIDATION FOR HYPERPARAMETERS OPTIMIZATION


HPo = grid_search_ttt(DATAhpo,HP_gs,class_train,class_test);

%% ADD FIRST ELEMENT TO DICTIONARY

% Get statistics from data (For Video Function)
DATAn.Xmax = max(DATAttt.input,[],2);	% max value
DATAn.Xmin = min(DATAttt.input,[],2);	% min value
DATAn.Xmed = mean(DATAttt.input,2);     % mean value
DATAn.Xdp = std(DATAttt.input,[],2);	% std value

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

%% PRESEQUENTIAL (TEST-THAN-TRAIN)

for n = 2:Nttt,
    
    % Display number of samples already seen (for debug)
    
    if(mod(n,1000) == 0),
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data
    
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

x = 1:Nttt;

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