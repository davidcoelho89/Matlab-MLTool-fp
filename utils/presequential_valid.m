function [PVout] = presequential_valid(DATA,HP,class_train,class_test,PSp)

% --- Presequential Validation Function ---
%
%   [PVout] = presequential_valid(DATA,HP,class_train,class_test,PSp)
%
%   Input:
%       DATA.
%           input = Matrix of training attributes             	[p x N]
%           output = Matrix of training labels                 	[Nc x N]
%       HP = set of HyperParameters to be tested
%       class_train = handler for classifier's training function
%       class_test = handler for classifier's classification function       
%       PSp.
%           repetitions = number of times the data is           	[cte]
%                         presented to the algorithm
%           cost = Which cost function will be used                 [cte]
%               1: Error (any classifier)
%               2: Error and dictionary size (prototype based)
%               3: Error and number of SV (SVC based)
%               4: Error and number of neurons (NN based)
%           lambda = trade-off between error and other parameters  	[cte]
%   Output:
%       PSout.
%           measure = measure to be minimized
%           acc = mean accuracy for data set and parameters
%           err = mean error for data set and parameters
%           Ds = percentage of prototypes compared to the dataset
%           Nneurons = number of neurons (NN based)
%           Nsv = number of support vectors (SVC based)

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 4) || (isempty(PSp)))
    PSp.repetitions = 1; % 1 repetition
    PSp.cost = 1;        % Error (any classifier)
    PSp.lambda = 0.5;    % More weight for dictionary size
else
    if (~(isfield(PSp,'repetitions')))
        PSp.repetitions = 1;
    end
    if (~(isfield(PSp,'cost')))
        PSp.cost = 1;
    end
    if (~(isfield(PSp,'lambda')))
        PSp.lambda = 2;
    end
end

%% INITIALIZATIONS

% Get Data 

[~,N] = size(DATA.input);       % Number of samples

% Get Hyperparameters

repetitions = PSp.repetitions;	% Number of repetitions of algorithm
cost = PSp.cost;                % Which cost function will be used
lambda = PSp.lambda;            % trade-off between error and other par

% Init Outupts

accuracy = 0;                   % Init accurary
Ds = 0;                         % Init # prototypes (dictionary size)
Nneurons = 0;                   % Init # of neurons (for ANNs)
Nsv = 0;                        % Init # of support vectors (for SVMs)

% Init Parameters

DATAn.input = DATA.input(:,1);
DATAn.output = DATA.output(:,1);
PAR = class_train(DATAn,HP);

%% ALGORITHM

for repetition = 1:repetitions
    
    % Restrictions 1: About combination of hyperparameters
    
    restriction1 = restriction_hp(PAR,class_train);
    if(restriction1)
        break;
    end
    
    for n = 1:N
        
        % Restrictions 2: About combination of parameters during training

        restriction2 = restriction_par_training(PAR,class_train);
        if (cost == 2)
            if (restriction2)
                break;
            end
        end
        
        % Get current data
        
        DATAn.input = DATA.input(:,n);
        DATAn.output = DATA.output(:,n);
        [~,y_lbl] = max(DATAn.output);
        
        % Test (classify arriving data with current model)
        
        OUTn = class_test(DATAn,PAR);
        [~,yh_lbl] = max(OUTn.y_h);
        
        % Update Classification Accuracy
        
        if(y_lbl == yh_lbl)
            accuracy = accuracy + 1;
        end
        
        % Train (update model with arriving data)
        
        PAR = class_train(DATAn,PAR);
        
    end
    
end

% Get accuracy and error

accuracy = accuracy / (N * repetitions);
error = 1 - accuracy;

% Restriction 3: About combination of parameters after training

restriction3 = restriction_par_final(DATA,PAR,class_train);

% Generate measure (value to be minimized)

if (cost == 1)
    
    if(restriction1 || restriction2 || restriction3)
        measure = 1;         % Maximum Error
    else
        measure = error;     % Error Measure
    end
    
elseif(cost == 2)
    
    % Get Dictionary Size
    [~,Nk] = size(PAR.Cx);
    Ds = Ds + Nk / N;
    
    if(restriction1 || restriction2 || restriction3)
        measure = 1 + lambda;            % Maximum value
    else
        measure = Ds + lambda * error;   % Measure
    end
    
end

%% FILL OUTPUT STRUCTURE

PVout.PAR = PAR;            % Model parameters
PVout.measure = measure;    % Metric using chosen hyperparemeters
PVout.acc = accuracy;       % Accuracy of trained model
PVout.err = error;          % Error of trained model
PVout.Ds = Ds;              % Dictionary Size
PVout.Nneurons = Nneurons;  % Number of neurons
PVout.Nsv = Nsv;            % Number of support vectors

%% END