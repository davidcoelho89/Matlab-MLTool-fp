function [PVout] = presequential_valid(DATA,HP,f_train,f_class,PSp)

% --- Presequential Validation Function ---
%
%   [PVout] = presequential_valid(DATA,HP,f_train,f_class,PSp)
%
%   Input:
%       DATA.
%           input = Matrix of training attributes             	[p x N]
%           output = Matrix of training labels                 	[Nc x N]
%       HP = set of HyperParameters to be tested
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%       PSp.
%           iterations = number of times the data is                [cte]
%                        presented to the algorithm
%           type = type of presequential validation                 [cte]
%               1: takes into account just accurary
%               2: takes into account also the dicitionary size
%           lambda = trade-off between error and dictionary size    [0 - 1]
%   Output:
%       PSout.
%           metric = measure to be minimized
%           acc = mean accuracy for data set and parameters
%           err = mean error for data set and parameters
%           Ds = percentage of prototypes compared to the dataset

%% INITIALIZATIONS

% Get Data 

[~,N] = size(DATA.input);       % Number of samples

% Get Hyperparameters

if (nargin == 4)
    PSp.iterations = 1;         % 1 repetition
    PSp.type = 1;               % Not prototype-based algorithm
    PSp.lambda = 0.5;           % More weight for dictionary size
end

iterations = PSp.iterations;	% Number of repetitions of algorithm
type = PSp.type;                % If classifier is prototype-based or not
lambda = PSp.lambda;            % Trade-off between error and dict size

% Init Outupts

accuracy = 0;                   % Init accurary
Ds = 0;                         % Init # prototypes (dictionary size)

% Init Parameters

DATAn.input = DATA.input(:,1);
DATAn.output = DATA.output(:,1);
PAR = f_train(DATAn,HP);

%% ALGORITHM

for it = 1:iterations
    
    % Restrictions 1: About combination of hyperparameters
    
    restriction1 = gs_restricion_hp(PAR,f_train);
    if(restriction1)
        break;
    end
    
    for n = 1:N
        
        % Restrictions 2: About combination of parameters during training

        restriction2 = gs_restriction_par_training(PAR,f_train);
        if (type == 2)
            if (restriction2)
                break;
            end
        end
        
        % Get current data
        
        DATAn.input = DATA.input(:,n);
        DATAn.output = DATA.output(:,n);
        [~,y_lbl] = max(DATAn.output);
        
        % Test (classify arriving data with current model)
        
        OUTn = f_class(DATAn,PAR);
        [~,yh_lbl] = max(OUTn.y_h);
        
        % Update Classification Accuracy
        
        if(y_lbl == yh_lbl)
            accuracy = accuracy + 1;
        end
        
        % Train (update model with arriving data)
        
        PAR = f_train(DATAn,PAR);
        
    end
    
end

% Get accuracy and error

accuracy = accuracy / (N * iterations);
error = 1 - accuracy;

% Restriction 3: About combination of parameters after training

restriction3 = gs_restriction_par_final(DATA,PAR,f_train);

% Generate Metric (value to be minimized)

if (type == 1)
    
    if(restriction1 || restriction2 || restriction3)
        measure = 1;         % Maximum Error
    else
        measure = error;     % Error Measure
    end
    
elseif(type == 2)
    
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

%% END