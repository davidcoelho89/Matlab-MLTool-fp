%% Machine Learning ToolBox

% Rotating Hyperplane and k2nn classifier
% Author: David Nascimento Coelho
% Last Update: 2020/03/13

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 26;              % Which problem will be solved / used
OPT.norm = 0;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training
OPT.file = 'hyper_k2nn_p1.mat';     % file where all the variables will be saved

%% HYPERPARAMETERS - DEFAULT

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
HP.Von = 0;         % Enable / disable video 
HP.K = 1;           % Number of nearest neighbors (classify)
HP.Ktype = 3;       % Kernel Type
HP.sig2n = 0.001;   % Kernel Regularization parameter
HP.sigma = 2;    	% Kernel width (gaussian)
HP.gamma = 2;       % polynomial order (poly 2 or 3)
HP.alpha = 1;       % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 1;       % Dot product adding (poly 1 / sigm 0.1)

%% HIPERPARAMETERS - GRID FOR TRAINING AND TESTING

% Set Variables Hyperparameters

K2NNcv = HP;
K2NNcv.v1 = 2.^linspace(-13,6,20);
K2NNcv.gamma = [2,3];

% Number of repetitions of the algorithm

OPT.Nr = length(K2NNcv.v1)*length(K2NNcv.gamma);

% Variable HyperParameters

% % linear 1
% K2NNcv.v1 = 2.^linspace(-15,5,21);
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

DATA = data_class_loading(OPT);     % Load Data Set

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

[Nc,N] = size(DATA.output);        	% get number of classes and samples

%% ACCUMULATORS

accuracy_vector_acc = cell(OPT.Nr,1);
no_of_correct_acc = cell(OPT.Nr,1);
no_of_errors_acc = cell(OPT.Nr,1);
predict_vector_acc = cell(OPT.Nr,1);
no_of_samples_acc = cell(OPT.Nr,1);
no_of_prot_acc = cell(OPT.Nr,1);

PAR_acc = cell(OPT.Nr,1);

% figure; VID = struct('cdata',cell(1,N),'colormap', cell(1,N));

%% SEQUENTIAL TESTS AND STATISTICS

disp('Begin Algorithm');

r = 0;

for i = 1:length(K2NNcv.v1),
for j = 1:length(K2NNcv.gamma),
    
% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%
    
    r = r + 1;
    display(r);
    display(datestr(now));

% %%%%%%%%%%%%% UPDATE HYPERPARAMETERS %%%%%%%%%%%%%%%%%%%

    HP.v1 = K2NNcv.v1(i);
    HP.gamma = K2NNcv.gamma(j);

% %%%%%%%%%%%%%%%%%%% INIT VECTORS %%%%%%%%%%%%%%%%%%%%%%%

    accuracy_vector = zeros(1,N);       % Hold Acc / (Acc + Err)
    no_of_correct = zeros(1,N);         % Hold # of correctly classified x
    no_of_errors = zeros(1,N);          % Hold # of misclassified x
    predict_vector = zeros(2,N);        % Hold true and predicted labels
    no_of_samples = zeros(Nc,N);        % Hold # of samples per class
    no_of_prot = zeros(Nc,N);           % Hold # of prot per class

% %%%%%%%%%%%%%%%% TRAINING AND TEST %%%%%%%%%%%%%%%%%%%%%

    % Get first element to dictionary
    DATAn.input = DATA.input(:,1);      % first element input
    DATAn.output = DATA.output(:,1);    % first element output
    DATAn.Xmax = max(DATA.input,[],2);  % max value
    DATAn.Xmin = min(DATA.input,[],2);  % min value
    DATAn.Xmed = mean(DATA.input,2);    % mean value
    DATAn.Xdp = std(DATA.input,[],2);   % std value

    % add element to dictionary
    [~,max_y] = max(DATAn.output);
    no_of_samples(max_y,1) = 1;
    no_of_prot(max_y,1) = 1;
    PAR = k2nn_train(DATAn,HP);

    for n = 2:N,

        % Display number of samples already seen (for debug)

        if(mod(n,1000) == 0),
            disp(n);
            disp(datestr(now));
        end

        % Get current data

        DATAn.input = DATA.input(:,n);
        DATAn.output = DATA.output(:,n);

        % Test (classify arriving data with current model)

        OUTn = k2nn_classify(DATAn,PAR);

        % Statistics

        [~,max_y] = max(DATAn.output);
        [~,max_yh] = max(OUTn.y_h);

        predict_vector(1,n) = max_y;
        predict_vector(2,n) = max_yh;

        for c = 1:Nc,
            if (c == max_y),
                no_of_samples(c,n) = no_of_samples(c,n-1) + 1;
            else
                no_of_samples(c,n) = no_of_samples(c,n-1);
            end
        end

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

%         if (HP.Von),
%             VID(n) = prototypes_frame(PAR.Cx,DATAn);
%         end

        % Hold number of prototypes per class
        
        [~,prot_lbl] = max(PAR.Cy);
        for c = 1:Nc,
            no_of_prot(c,n) = sum(prot_lbl == c);
        end
        
        % Break if number of prototypes is too high
        
        [~,mt] = size(PAR.Cx);
        if (mt > 800),
            break;
        end

    end

    accuracy_vector_acc{r} = accuracy_vector;
    no_of_correct_acc{r} = no_of_correct;
    no_of_errors_acc{r} = no_of_errors;
    predict_vector_acc{r} = predict_vector;
    no_of_samples_acc{r} = no_of_samples;
    no_of_prot_acc{r} = no_of_prot;
    PAR_acc{r} = PAR;

end
end

%% PLOTS

x = 1:N;

% Data and Prototypes
figure;
hold on 
plot(DATA.input(1,:),DATA.input(2,:),'r.');
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
title('Data and Prototypes');
hold off

% Number of hits x number of errors
figure;
hold on
plot(x,no_of_errors,'r-');
plot(x,no_of_correct,'b-');
title('number of hits and errors');
hold off

% Percentage of Correct Classified
figure;
plot(x,accuracy_vector,'r-');
title('number of correct classified')

% Number of samples per class
figure;
colors = lines(Nc);
hold on
for c = 1:Nc,
    plot(x,no_of_samples(c,:),'Color',colors(c,:));
end
title('number of samples per class');
hold off

% Number of prototypes per class
figure;
colors = lines(Nc);
hold on
for c = 1:Nc,
    plot(x,no_of_prot(c,:),'Color',colors(c,:));
end
title('number of prototypes per class');

%% SAVE FILE

save(OPT.file);

%% END