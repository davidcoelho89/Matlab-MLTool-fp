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

OPT.prob = 25;              % Which problem will be solved / used
OPT.prob2 = 30;             % More details about a specific data set
OPT.norm = 0;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 50;              	% Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% HYPERPARAMETERS - DEFAULT

HP.Dm = 2;          % Design Method
HP.Ss = 1;          % Sparsification strategy
HP.v1 = 0.8;        % Sparseness parameter 1 
HP.v2 = 0.9;        % Sparseness parameter 2
HP.Us = 0;          % Update strategy
HP.eta = 0.01;      % Update rate
HP.Ps = 0;          % Prunning strategy
HP.min_score = -10; % Score that leads the sample to be pruned
HP.max_prot = Inf;  % Max number of prototypes
HP.min_prot = 1;    % Min number of prototypes
HP.Von = 1;         % Enable / disable video 
HP.K = 1;           % Number of nearest neighbors (classify)
HP.knn_type = 2;    % Type of knn aproximation
HP.Ktype = 2;       % Kernel Type
HP.sig2n = 0.001;   % Kernel Regularization parameter
HP.sigma = 2;    	% Kernel width (gauss, exp, cauchy, log, kmod)
HP.gamma = 2;       % polynomial order (poly 2 or 3)
HP.alpha = 0.1;     % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 0.1;     % Dot product adding (poly 1 / sigm 0.1)

%% HYPERPARAMETERS - FOR GRID SEARCH

HP_gs = HP;
HP_gs.v1 = 2.^linspace(-10,10,21);

% Kernel Functions: 1 lin    / 2 gauss / 3 poly / 4 exp  /
%                   5 cauchy / 6 log   / 7 sigm / 8 kmod /

% % linear 1
% HP_gs.v1 = 2.^linspace(-10,10,21);
% % Gaussian 2
% HP_gs.v1 = 2.^linspace(-4,3,8);
% HP_gs.sigma = 2.^linspace(-10,9,20);
% % Polynomial 3
% HP_gs.v1 = 2.^linspace(-13,6,20);
% HP_gs.gamma = [2,3];
% % Exponential 4
% HP_gs.v1 = 2.^linspace(-4,3,8);
% HP_gs.sigma = 2.^linspace(-10,9,20);
% % Cauchy 5
% HP_gs.v1 = 2.^linspace(-4,3,8);
% HP_gs.sigma = 2.^linspace(-10,9,20);
% % Log 6
% HP_gs.v1 = -2.^linspace(10,2,9);
% HP_gs.sigma = [0.001 0.01 0.1 1 2 5];
% % Sigm 7
% HP_gs.v1 = 2.^linspace(-13,6,20);
% HP_gs.alpha = 2.^linspace(-8,2,11);
% HP_gs.theta = 2.^linspace(-8,2,11);

%% DATA LOADING AND PRE-PROCESSING

% Load Dataset and Adjust its Labels

DATA = data_class_loading(OPT);     % Load Data Set
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

% Set data for the cross validation step: min (0.2 * N, 1000)

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

% Get statistics from data (For Video Function)
DATAn.Xmax = max(DATAttt.input,[],2);
DATAn.Xmin = min(DATAttt.input,[],2);
DATAn.Xmed = mean(DATAttt.input,2);
DATAn.Xdp = std(DATAttt.input,[],2);

%% DATA VISUALIZATION

figure; plot_data_pairplot(DATAttt);

%% ACCUMULATORS

samples_per_class = zeros(Nc,Nttt);	% Hold number of samples per class

accuracy_vector = zeros(1,Nttt);	% Hold Acc / (Acc + Err)
no_of_correct = zeros(1,Nttt);      % Hold # of correctly classified x
no_of_errors = zeros(1,Nttt);       % Hold # of misclassified x

predict_vector = zeros(Nc,Nttt);	% Hold predicted labels

prot_per_class = zeros(Nc+1,Nttt);	% Hold number of prot per class
                                    % Last is for the sum
                                    
VID = struct('cdata',cell(1,Nttt),'colormap', cell(1,Nttt));

%% CROSS VALIDATION FOR HYPERPARAMETERS OPTIMIZATION

display('begin grid search')

HPo = grid_search_ttt(DATAhpo,HP_gs,@isk2nn_train,@isk2nn_test);

%% PRESEQUENTIAL (TEST-THAN-TRAIN)

display('begin Test-than-train')

% Add first element to dictionary

DATAn.input = DATA.input(:,1);      % First element input
DATAn.output = DATA.output(:,1);    % First element output
[~,max_y] = max(DATAn.output);      % Get sample's class
samples_per_class(max_y,1) = 1;     % Update number of samples per class
PAR = k2nn_train(DATAn,HPo);     	% Add element

for n = 2:Nttt,
    
    % Display number of samples already seen (for debug)
    
    if(mod(n,1000) == 0),
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data
    
    DATAn.input = DATA.input(:,n);
    DATAn.output = DATA.output(:,n);
    [~,y_lbl] = max(DATAn.output);
    
    % Test (classify arriving data with current model)
    % Train (update model with arriving data)
        
    PAR = isk2nn_train(DATAn,PAR);
    predict_vector(:,n) = PAR.y_h;
    [~,yh_lbl] = max(PAR.y_h);
    
    % Hold Number of Samples per Class 
    
    for c = 1:Nc,
        if (c == y_lbl),
            samples_per_class(c,n) = samples_per_class(c,n-1) + 1;
        else
            samples_per_class(c,n) = samples_per_class(c,n-1);
        end
    end
    
    % Hold Number of Errors and Hits

    if (y_lbl == yh_lbl),
        no_of_correct(n) = no_of_correct(n-1) + 1;
        no_of_errors(n) = no_of_errors(n-1);
    else
        no_of_correct(n) = no_of_correct(n-1);
        no_of_errors(n) = no_of_errors(n-1) + 1;
    end
    accuracy_vector(n) = no_of_correct(n) / ...
                        (no_of_correct(n) + no_of_errors(n));
    
    % Hold Number of prototypes per Class
    
    [~,lbls] = max(PAR.Cy);
    for c = 1:Nc,
        prot_per_class(c,n) = sum(lbls == c);
    end
    [~,Nprot] = size(PAR.Cy);
    prot_per_class(Nc+1,n) = Nprot;
    
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

% Number of samples per class
figure;
colors = lines(Nc);
hold on
for c = 1:Nc,
    plot(x,samples_per_class(c,:),'Color',colors(c,:));
end
hold off

% Number of Prototypes (Total and per class)
figure;
colors = lines(Nc+1);
hold on
for c = 1:Nc+1,
    plot(x,prot_per_class(c,:),'Color',colors(c,:));
end

% Number of hits x number of errors
figure;
hold on
plot(x,no_of_errors,'r-');
plot(x,no_of_correct,'b-');
hold off

% Percentage of Correct Classified
figure;
plot(x,accuracy_vector,'r-');

%% SAVE FILE



%% END