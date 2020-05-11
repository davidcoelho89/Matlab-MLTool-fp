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
%           type = type of cross validation                         [cte]
%               1: takes into account just accurary
%               2: takes into account also the dicitionary size
%           lambda = trade-off between error and dictionary size    [0 - 1]
%   Output:
%       PSout.
%           metric = metric to be minimized
%           acc = mean accuracy for data set and parameters
%           err = mean error for data set and parameters
%           Ds = percentage of prototypes compared to the dataset

%% INITIALIZATIONS

% Get Data 

[~,N] = size(DATA.input);       % Number of samples
[Nc,~] = size(DATA.output);     % Number of classes

% Get Hyperparameters

if (nargin == 4),
    PSp.iterations = 1;
    PSp.type = 1;
    PSp.lambda = 0.5;
end

type = PSp.type;                % If classifier is prototype-based or not
lambda = PSp.lambda;            % Trade-off between error and dict size
iterations = PSp.iterations;	% Number of iterations

% Init Parameters

if (~(isfield(HP,'max_prot'))),
    HP.max_prot = Inf;
end

% Init Outupts

accuracy = 0;                   % Init accurary
Ds = 0;                         % Init # prototypes (dictionary size)

% Add first Element to dictionary

DATAn.input = DATA.input(:,1);
DATAn.output = DATA.output(:,1);
PAR = f_train(DATAn,HP);

%% ALGORITHM

for it = 1:iterations
    
    for n = 1:N,
        
        % Get current data
        
        DATAn.input = DATA.input(:,n);
        DATAn.output = DATA.output(:,n);
        [~,y_lbl] = max(DATAn.output);
        
        % Test (classify arriving data with current model)
        
        OUTn = f_class(DATAn,PAR);
        [~,yh_lbl] = max(OUTn.y_h);
        
        % Update Classification Accuracy
        
        if(y_lbl == yh_lbl),
            accuracy = accuracy + 1;
        end
        
        % Train (update model with arriving data)
        
        PAR = f_train(DATAn,PAR);
        
        % Restriction: max number of prototypes (prototype based models)
        
        if (type == 2),
            [~,Nk] = size(PAR.Cx);
            if (Nk >= PAR.max_prot),
                break;
            end
        end
        
    end
    
end

% Get accuracy and error

accuracy = accuracy / (N * iterations);
error = 1 - accuracy;

% Generate Metric (value to be minimized)

if (type == 1),

    metric = error;

elseif(type == 2),

    % Get Dictionary Size
    [~,Nk] = size(PAR.Cx);
    Ds = Ds + Nk / N;
    
    if (Nk <= Nc || Nk >= PAR.max_prot),
        metric = 1 + lambda;    % Maximum value
    else
        metric = Ds + lambda * error;
    end
    
end

%% FILL OUTPUT STRUCTURE

PVout.PAR = PAR;        % Model parameters
PVout.metric = metric;  % Metric using chosen hyperparemeters
PVout.acc = accuracy;   % Accuracy of trained model
PVout.err = error;      % Error of trained model
PVout.Ds = Ds;          % 

%% END