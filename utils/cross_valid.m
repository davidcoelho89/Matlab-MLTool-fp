function [CVout] = cross_valid(DATAtr,HP_probe,class_train,class_test,CVp)

% --- Cross Validation Function ---
%
%   [CVout] = cross_valid(DATAtr,HP_probe,class_train,class_test,CVp)
%
%   Input:
%       DATAtr.
%           input = Matrix of training attributes                       [p x N]
%           output = Matrix of training labels                          [Nc x N]
%       HP_probe = set of HyperParameters to be tested                  [struct]
%       class_train = handler for classifier's training function
%       class_test = handler for classifier's test function       
%       CVp.
%           fold = number of data partitions                            [cte]
%           cost = Which cost function will be used                     [cte]
%               1: Error (any classifier)
%               2: Error and dictionary size (prototype based)
%               3: Error and number of SV (SVC based)
%               4: Error and number of neurons (NN based)
%               5: Error, dictionary size, f1-score
%           lambda = trade-off between error and other parameters  	    [cte]
%           gamma = trade-off between f1s (or mcc) and other parameters [cte]
%   Output:
%       CVout.
%           measure = measure to be minimized
%           acc = mean accuracy for data set and parameters
%           err = mean error for data set and parameters
%           Ds = percentage of prototypes compared to the dataset
%           Nneurons = number of neurons (NN based)
%           Nsv = number of support vectors (SVC based)
%           fsc = F1-score
%           mcc = Matthews Correlation Coefficient

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 4) || (isempty(CVp)))
    CVp.fold = 5;           % 5 folds
    CVp.cost = 1;           % Error (any classifier)
    CVp.lambda = 2;         % More weight for error
    CVp.gamma = 0.1;        % Small weight for mcc or f1s
else
    if (~(isfield(CVp,'fold')))
        CVp.fold = 5;
    end
    if (~(isfield(CVp,'cost')))
        CVp.cost = 1;
    end
    if (~(isfield(CVp,'lambda')))
        CVp.lambda = 2;
    end
    if (~(isfield(CVp,'gamma')))
        CVp.gamma = 0.1;
    end
end

%% INITIALIZATIONS

% Get Data 

X = DATAtr.input;     	% Attributes Matrix [p x N]
Y = DATAtr.output;     	% labels Matriz [Nc x N]
[~,N] = size(X);      	% Number of samples

% Get HyperParameters

Nfold = CVp.fold;     	% Number of data partitions
part = floor(N/Nfold); 	% Size of each data partition
cost = CVp.cost;        % Which cost function will be used
lambda = CVp.lambda;    % trade-off between error and dictionary size

% Init Outupts

accuracy = 0;          	% Init accurary
Ds = 0;                 % Dictionary Size
Nneurons = 0;           % Init # of neurons (NN based)
Nsv = 0;                % Init # of support vectors (SVC based)
fsc = 0;                % Init F1-score measure
mcc = -1;               % Init Matthews Correlation Coefficient

%% ALGORITHM

for fold = 1:Nfold
    
    % Restriction 1: About combination of hyperparameters
    
    restriction1 = restriction_hp(HP_probe,class_train);
    if(restriction1)
        break;
    end

    % Define Data division

    if fold == 1
        DATAtr.input  = X(:,part+1:end);
        DATAtr.output = Y(:,part+1:end);
        DATAts.input  = X(:,1:part);
        DATAts.output = Y(:,1:part);
    elseif fold == Nfold
        DATAtr.input  = X(:,1:(Nfold-1)*part);
        DATAtr.output = Y(:,1:(Nfold-1)*part);
        DATAts.input  = X(:,(Nfold-1)*part+1:end);
        DATAts.output = Y(:,(Nfold-1)*part+1:end);
    else
        DATAtr.input  = [X(:,1:(fold-1)*part) X(:,fold*part+1:end)];
        DATAtr.output = [Y(:,1:(fold-1)*part) Y(:,fold*part+1:end)];
        DATAts.input  = X(:,(fold-1)*part+1:fold*part);
        DATAts.output = Y(:,(fold-1)*part+1:fold*part);
    end

    % Training of classifier
    [PAR] = class_train(DATAtr,HP_probe);
    
    % Uses just accuracy as measure
    if (cost == 1)
        
    % Accumulate Number of Prototypes (for prototype-based classifiers)
    elseif (cost == 2)
        [~,Nk] = size(PAR.Cx);
        Ds = Ds + Nk;
    
    % Accumulate Number of Support Vectors (for SV-based classifiers)
    elseif (cost == 3)
    
    % Accumulate Number of neurons (for NN-based classifiers)
    elseif (cost == 4)
        
    end

    % Test of classifier
    [OUT] = class_test(DATAts,PAR);

    % Statistics of Classifier
    [STATS_ts] = class_stats_1turn(DATAts,OUT);

    % Accumulate Accuracy rate
    accuracy = accuracy + STATS_ts.acc;
    
end

% Dont need to calculate restriction2 and/or error, if 
% there is a prohibited combination of hyperparameters
if(restriction1)
    restriction2 = 1;
    mean_error = 1;
    mean_accuracy = 0;
else
    
    mean_accuracy = accuracy / Nfold;
    mean_error = 1 - mean_accuracy;

    % Restriction 2: About combination of parameters after training
    restriction2 = restriction_par_final(DATAtr,PAR,class_train);    
    
end

% Generate measure (value to be minimized)

if (cost == 1)
    
    if(restriction1 || restriction2)
        measure = 1;        % Maximum Error
    else
        measure = mean_error;    % Error Measure
    end
    
elseif (cost == 2)
    
    % Get dictionary mean size (Mean Percentage of Prototypes)
    Ds = Ds / (N * Nfold);
    
    if(restriction1 || restriction2)
        measure = 1 + lambda;           % Maximum value
    else
        measure = Ds + lambda * mean_error;  % Measure
    end

elseif (cost == 3)
    % ToDo - all

elseif (cost == 4)
    % ToDo - all

elseif (cost == 5)
    % ToDo - all
    
end

%% FILL OUTPUT STRUCTURE

CVout.measure = measure;    % Metric using chosen hyperparemeters
CVout.acc = mean_accuracy;  % Accuracy of trained model
CVout.err = mean_error;     % Error of trained model
CVout.Ds = Ds;              % Dictionary Size
CVout.Nneurons = Nneurons;  % Number of neurons
CVout.Nsv = Nsv;            % Number of support vectors
CVout.fsc = fsc;            % F1-score measure
CVout.mcc = mcc;            % Matthews Correlation Coefficient Measure

%% END