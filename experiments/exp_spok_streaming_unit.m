%% Machine Learning ToolBox

% Sample used to run spok Algorithm and a Streaming DataSet
% Author: David Nascimento Coelho
% Last Update: 2020/05/11

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General Datasets / Problems:

% 25 - sea / 26 - Hyperplane / 27 - RBFmov / 28 - RBFint / 29 - Squares /
% 30 - Chess / 31 - MixDrift / 32 - led / 33 - weather / 34 - Electricity / 
% 35 - CoverType / 36 - Poker / 37 - Outdoor / 38 - Rialto / 39 - Spam /

% General options' structure

OPT.prob = 36;              % Which problem will be solved / used
OPT.prob2 = 30;             % More details about a specific data set
OPT.norm = 0;               % Normalization definition
OPT.lbl = 1;                % Labeling definition. 1: [-1 +1] pattern
OPT.Nr = 01;              	% Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% HYPERPARAMETERS - DEFAULT

HP.Dm = 2;                  % Design Method
HP.Ss = 1;                  % Sparsification strategy
HP.v1 = 0.8;                % Sparseness parameter 1 
HP.v2 = 0.9;                % Sparseness parameter 2
HP.Us = 1;                  % Update strategy
HP.eta = 0.10;              % Update rate
HP.Ps = 2;                  % Prunning strategy
HP.min_score = -10;         % Score that leads the sample to be pruned
HP.max_prot = 600;          % Max number of prototypes
HP.min_prot = 1;            % Min number of prototypes
HP.Von = 0;                 % Enable / disable video 
HP.K = 1;                   % Number of nearest neighbors (classify)
HP.knn_type = 2;            % Type of knn aproximation
HP.Ktype = 2;               % Kernel Type (2: Gaussian / see kernel_func())
HP.sig2n = 0.001;           % Kernel Regularization parameter
HP.sigma = 2;               % Kernel width (gauss, exp, cauchy, log, kmod)
HP.alpha = 0.1;             % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 0.1;             % Dot product adding (poly 1 / sigm 0.1)
HP.gamma = 2;               % polynomial order (poly 2 or 3)

%% HYPERPARAMETERS - FOR OPTIMIZATION

HP_gs = HP;

HP_gs.v1 = 2.^linspace(-4,3,8);
HP_gs.v2 = HP_gs.v1(end) + 0.001;
HP_gs.sigma = 2.^linspace(-10,9,20);

%% DATA LOADING AND PRE-PROCESSING

% Load Dataset and Adjust its Labels

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

% Set data for the cross validation step: min (0.2 * N, 1000)

if (N < 5000)
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

% Get Normalization Parameters
PARnorm = normalize_fit(DATAhpo,OPT);

% Normalize all data
DATA = normalize_transform(DATA,PARnorm);

% Normalize hpo data
DATAhpo = normalize_transform(DATAhpo,PARnorm);

% Normalize ttt data
DATAttt = normalize_transform(DATAttt,PARnorm);

% Get statistics from data (For Video Function)
DATAn.Xmax = max(DATA.input,[],2);
DATAn.Xmin = min(DATA.input,[],2);
DATAn.Xmed = mean(DATA.input,2);
DATAn.Xstd = std(DATA.input,[],2);

%% DATA VISUALIZATION

% plot_data_pairplot(DATAhpo);        % See pairplot of attributes

% DATA1.input = DATA.input(:,4001:5000);
% DATA1.output = DATA.output(:,4001:5000);
% plot_data_pairplot(DATA1);

%% ACCUMULATORS

samples_per_class = zeros(Nc,Nttt);	% Hold number of samples per class

predict_vector = zeros(Nc,Nttt);	% Hold predicted labels

no_of_correct = zeros(1,Nttt);      % Hold # of correctly classified x
no_of_errors = zeros(1,Nttt);       % Hold # of misclassified x

accuracy_vector = zeros(1,Nttt);	% Hold Acc / (Acc + Err)

prot_per_class = zeros(Nc+1,Nttt);	% Hold number of prot per class
                                    % Last is for the sum
                                    
VID = struct('cdata',cell(1,Nttt),'colormap', cell(1,Nttt));

%% GRID SEARCH FOR HYPERPARAMETERS OPTIMIZATION

disp('begin grid search')

% Grid Search Parameters

GSp.iterations = 01; % number of times data is presented to the algorithm
GSp.type = 2;        % Takes into account also the dicitionary size
GSp.lambda = 2; 	 % Jpbc = Ds + lambda * Err

% Get Hyperparameters Optimized and the Prototypes Initialized

PAR = grid_search_ttt(DATAhpo,HP_gs,@spok_train,@spok_classify,GSp);

% PAR.max_prot = 1000;

%% PRESEQUENTIAL (TEST-THAN-TRAIN)

disp('begin Test-than-train')

figure; % new figure for video ploting

for n = 1:Nttt
    
    % Display number of samples already seen (for debug)
    
    if(mod(n,1000) == 0)
        disp(n);
        disp(datestr(now));
    end
    
    % Get current data
    
    DATAn.input = DATAttt.input(:,n);
    DATAn.output = DATAttt.output(:,n);
    [~,y_lbl] = max(DATAn.output);
    
    % Test  (classify arriving data with current model)
    % Train (update model with arriving data)
    
    PAR = spok_train(DATAn,PAR);
    
    % Hold Number of Samples per Class 
    
    if n == 1
        samples_per_class(y_lbl,n) = 1; % first element
    else
        samples_per_class(:,n) = samples_per_class(:,n-1);
        samples_per_class(y_lbl,n) = samples_per_class(y_lbl,n-1) + 1;
    end
    
    % Hold Predicted Labels
    
    predict_vector(:,n) = PAR.y_h;
    [~,yh_lbl] = max(PAR.y_h);
    
    % Hold Number of Errors and Hits
    
    if n == 1
        if (y_lbl == yh_lbl)
            no_of_correct(n) = 1;
        else
            no_of_errors(n) = 1;
        end
    else
        if (y_lbl == yh_lbl)
            no_of_correct(n) = no_of_correct(n-1) + 1;
            no_of_errors(n) = no_of_errors(n-1);
        else
            no_of_correct(n) = no_of_correct(n-1);
            no_of_errors(n) = no_of_errors(n-1) + 1;
        end
    end
    
    % Hold Accuracy
    
    accuracy_vector(n) = no_of_correct(n) / ...
                        (no_of_correct(n) + no_of_errors(n));
    
    % Hold Number of prototypes per Class
    
    [~,lbls] = max(PAR.Cy);
    for c = 1:Nc
        prot_per_class(c,n) = sum(lbls == c);
    end
    
    [~,Nprot] = size(PAR.Cy);
    prot_per_class(Nc+1,n) = Nprot;
    
    % Video Function
    
    if (HP.Von)
        VID(n) = prototypes_frame(PAR.Cx,DATAn);
    end
    
end

%% STATS

OUT.y_h = predict_vector;
STATS = class_stats_1turn(DATAttt,OUT);

%% PLOTS

x = 1:Nttt;

% Data and Prototypes
figure;
hold on
plot(DATAttt.input(1,:),DATAttt.input(2,:),'r.');
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
title('Prototypes and Data')
xlabel('Attribute 1')
ylabel('Attribute 2')
hold off

% Number of samples per class
figure;
colors = lines(Nc);
hold on
for c = 1:Nc
    plot(x,samples_per_class(c,:),'Color',colors(c,:));
end
title('Number of Samples Per Class')
hold off

% Number of Prototypes (Total and per class)
figure;
colors = lines(Nc+1);
hold on
% for c = 1:Nc+1,
%     plot(x,prot_per_class(c,:),'Color',colors(c,:));
% end
plot(x,prot_per_class(Nc+1,:),'Color',colors(Nc+1,:));
title('Number of Prototypes per step')
xlabel('Steps')
ylabel('Number of Prototypes')
hold off

% Number of hits x number of errors
figure;
hold on
plot(x,no_of_errors,'r-');
plot(x,no_of_correct,'b-');
title('number of hits and errors')
hold off

% Percentage of Misclassified
figure;
hold on
plot(x,1-accuracy_vector,'r-');
title('Percentage of samples misclassified')
xlabel('Time step')
ylabel('Error Rate')
axis([-1 length(x) -0.1 1.1])
hold off

%% SAVE FILE

% save(OPT.file,'-v7.3')/

%% END