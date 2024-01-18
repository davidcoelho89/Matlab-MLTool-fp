function [HPoptm] = grid_search_cv(DATAtr,HPgs,class_train,class_test,CVp)

% --- Optm hyperparameters definition by Grid Search and Cross Validation ---
%
%   [HPoptm] = grid_search_cv(DATAtr,HPgs,class_train,class_test,CVp)
%
%   Input:
%       DATAtr.
%           input = training attributes                                 [p x N]
%           output = training labels                                    [Nc x N]
%       HPgs = hyperparameters for grid search                          [struct]
%              (vectors or cells containing values that will be tested)
%       class_train = handler for classifier's training function
%       class_test = handler for classifier's test function       
%       CVp.
%           fold = number of data partitions for cross validation	    [cte]
%           cost = Which cost function will be used                     [cte]
%               1: Error (any classifier)
%               2: Error and dictionary size (prototype based)
%               3: Error and number of SV (SVC based)
%               4: Error and number of neurons (NN based)
%               5: Error, dictionary size, f1-score
%           lambda = trade-off between error and other parameters       [cte]
%           gamma = trade-off between f1s (or mcc) and other parameters [cte]
%   Output:
%       HPoptm = Optimum hyperparameters of classifier for dataset      [struct]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 4) || (isempty(CVp)))
    CVp.fold = 5;
    CVp.cost = 1;
    CVp.lambda = 0.5;
    CVp.gamma = 0.1;
else
    if (~(isfield(CVp,'fold')))
        CVp.fold = 5;
    end
    if (~(isfield(CVp,'cost')))
        CVp.cost = 1;
    end
    if (~(isfield(CVp,'lambda')))
        CVp.lambda = 0.5;
    end
    if (~(isfield(CVp,'gamma')))
        CVp.gamma = 0.1;
    end
end

%% INITIALIZATIONS

% Get General Characteristics of Problem

HP_names = fieldnames(HPgs);            % Names of HP
HP_number = numel(HP_names);            % Number of HP
HP_index = ones(HP_number,1);           % Index for each HP in grid search
HP_index_max = zeros(HP_number,1);      % Max index for each HP in grid search

% Init Max Index of HyperParameters

for i = 1:HP_number
    HP_name = HP_names{i};                  % Get HP name
    HP_values = HPgs.(HP_name);             % Get HP values vector
    HP_index_max(i) = length(HP_values);	% Get HP values vector length
end

% Init Auxiliary Variables

max_iterations = prod(HP_index_max);

%% ALGORITHM

for turn = 1:max_iterations
    
    % Define Hyperparameters that will be tested
    
    for j = 1:HP_number
        HP_name = HP_names{j};                              % Get HP name
        HP_values = HPgs.(HP_name);                         % Get HP values
        if(iscell(HP_values))
            HP_probe.(HP_name) = HP_values{HP_index(j)};	% Get HP value
        else
            HP_probe.(HP_name) = HP_values(HP_index(j));	% Get HP value
        end
    end
    
    % Cross Validation
    
    CVout = cross_valid(DATAtr,HP_probe,class_train,class_test,CVp);

    % Define New Optimum HyperParameters
    
    if (turn == 1)
        HPoptm = HP_probe;
        min_measure = CVout.measure;
    else
        if (CVout.measure < min_measure)
            HPoptm = HP_probe;
            min_measure = CVout.measure;
        end
    end

    % Grid Search - Update indexes of HP (uses "digital clock logic")

    i = 1;
    while i <= HP_number
        HP_index(i) = HP_index(i)+1;
        if HP_index(i) > HP_index_max(i)
            HP_index(i) = 1;
            i = i + 1;
        else
            break;
        end
    end
    
end

%% END