function [HPoptm] = random_search_ttt(DATA,HPgs,class_train,class_test,PSp)

% --- Optm hyperparameters definition by Random Search for Seq. Learn ---
%
%   [HPoptm] = random_search_ttt(DATA,HPgs,class_train,class_test,PSp)
%
%   Input:
%       DATA.
%           input = training attributes                                 [p x N]
%           output = training labels                                    [Nc x N]
%       HPgs = hyperparameters for random search                        [struct]
%              (vectors or cells containing values that can be tested)
%       class_train = handler for classifier's training function
%       class_test = handler for classifier's classification function       
%       PSp.
%           max_it = maximum number of iterations                       [cte]
%           repetitions = number of times the data is                   [cte]
%                        presented to the algorithm
%           cost = Which cost function will be used                     [cte]
%               1: Error (any classifier)
%               2: Error and dictionary size (prototype based)
%               3: Error and number of SV (SVC based)
%               4: Error and number of neurons (NN based)
%               5: Error, dictionary size, f1-score
%           lambda = trade-off between error and other parameters       [cte]
%           gamma = trade-off between f1s (or mcc) and other parameters [cte]
%   Output:
%       HPoptm = optimum hyperparameters of classifier for data set

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 4) || (isempty(PSp)))
    PSp.max_it = 100;
    PSp.repetitions = 1;
    PSp.cost = 1;
    PSp.lambda = 0.5;
    PSp.gamma = 0.1;
else
    if (~(isfield(PSp,'max_it')))
        PSp.max_it = 100;
    end
    if (~(isfield(PSp,'repetitions')))
        PSp.repetitions = 1;
    end
    if (~(isfield(PSp,'cost')))
        PSp.cost = 1;
    end
    if (~(isfield(PSp,'lambda')))
        PSp.lambda = 0.5;
    end
    if (~(isfield(PSp,'gamma')))
        PSp.gamma = 0.1;
    end
end

%% INITIALIZATIONS

% Get General Characteristics of Problem

HP_names = fieldnames(HPgs);      	% Names of HP
HP_number = numel(HP_names);    	% Number of HP
HP_index_max = zeros(HP_number,1);	% Max index for each HP in grid search

% Init Max Index of HyperParameters

for i = 1:HP_number
    HP_name = HP_names{i};                  % Get HP name
    HP_values = HPgs.(HP_name);             % Get HP values vector
    HP_index_max(i) = length(HP_values);	% Get HP values vector length
end

% Init Auxiliary Variables

indexes_matrix = rand(HP_number,PSp.max_it);
for i = 1:HP_number
    indexes_matrix(i,:) = ceil(indexes_matrix(i,:)*HP_index_max(i));
end

% Debug: verify values of randomly generated indexes

% histogram(indexes_matrix(HP_number,:));

%% ALGORITHM

for turn = 1:PSp.max_it
    
    % Define Hyperparameters that will be tested
    
    for i = 1:HP_number
        HP_name = HP_names{i};                  % Get HP name
        HP_values = HPgs.(HP_name);             % Get HP values vector
        if(iscell(HP_values))
            HP_probe.(HP_name) = HP_values{indexes_matrix(i,turn)};
        else
            HP_probe.(HP_name) = HP_values(indexes_matrix(i,turn));	
        end
    end
    
    % "Interleaved Test-Then-Train" or "Prequential" Method
    
    PSout = presequential_valid(DATA,HP_probe,class_train,class_test,PSp);
    
    % Define New Optimum HyperParameters
    
    if (turn == 1)
        HPoptm = PSout.PAR;
        min_measure = PSout.measure;
    else
        if (PSout.measure < min_measure)
            HPoptm = PSout.PAR;
            min_measure = PSout.measure;
        end
    end
    
end

%% END