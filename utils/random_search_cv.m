function [HPoptm] = random_search_cv(DATAtr,HPgs,class_train,class_test,CVp)

% --- Optm hyperparameters definition by Random Search and Cross Valid ---
%
%   [HPoptm] = random_search_cv(DATAtr,HPgs,class_train,class_test,CVp)
%
%   Input:
%       DATAtr.
%           input = training attributes                            [p x N]
%           output = training labels                               [Nc x N]
%       HPgs = hyperparameters for random search                   [struct]
%              (vectors or cells containing values that can be tested)
%       class_train = handler for classifier's training function
%       class_test = handler for classifier's test function       
%       CVp.
%           max_it = maximum number of iterations                   [cte]
%           fold = number of data partitions for cross validation	[cte]
%           cost = Which cost function will be used                 [cte]
%               1: Error (any classifier)
%               2: Error and dictionary size (prototype based)
%               3: Error and number of SV (SVC based)
%               4: Error and number of neurons (NN based)
%           lambda = trade-off between error and other parameters  	[cte]
%   Output:
%       HPoptm = Optimum hyperparameters of classifier for dataset [struct]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 4) || (isempty(CVp)))
    CVp.max_it = 10;
    CVp.fold = 5;
    CVp.cost = 1;
    CVp.lambda = 0.5;
else
    if (~(isfield(CVp,'max_it')))
        CVp.max_it = 10;
    end
    if (~(isfield(CVp,'fold')))
        CVp.fold = 5;
    end
    if (~(isfield(CVp,'cost')))
        CVp.cost = 1;
    end
    if (~(isfield(CVp,'lambda')))
        CVp.lambda = 0.5;
    end
end

%% INITIALIZATIONS

% Get General Characteristics of Problem

HP_names = fieldnames(HPgs);       	% Names of HP
HP_number = numel(HP_names);    	% Number of HP
HP_index_max = zeros(HP_number,1);	% Max index for each HP in grid search

% Init Max Index of HyperParameters

for i = 1:HP_number
    HP_name = HP_names{i};                  % Get HP name
    HP_values = HPgs.(HP_name);             % Get HP values vector
    HP_index_max(i) = length(HP_values);	% Get HP values vector length
end

% Init Auxiliary Variables

indexes_matrix = rand(HP_number,CVp.max_it);
for i = 1:HP_number
    indexes_matrix(i,:) = ceil(indexes_matrix(i,:)*HP_index_max(i));
end

% Debug: verify values of randomly generated indexes

% histogram(indexes_matrix(HP_number,:));

%% ALGORITHM

for turn = 1:CVp.max_it
    
    % Define Hyperparameters that will be tested
    
    for i = 1:HP_number
        HP_name = HP_names{i};
        HP_values = HPgs.(HP_name);
        if(iscell(HP_values))
            HP_probe.(HP_name) = HP_values{indexes_matrix(i,turn)};
        else
            HP_probe.(HP_name) = HP_values(indexes_matrix(i,turn));	
        end
    end

%     for i = 1:HP_number
%         
%         HP_name = HP_names{i};                  % Get HP name
%         HP_values = HPgs.(HP_name);             % Get HP values vector
%         
%         if(HP_index_max(i)) == 1
%             index = 1;
%         else
%             ind_rand = randperm(HP_index_max(i));   % Get rand indexes
%             index = ind_rand(1);                    % Get first rand index
%         end
%         
%         if(iscell(HP_values))
%             HP_probe.(HP_name) = HP_values{index};	% Get HP value
%         else
%             HP_probe.(HP_name) = HP_values(index);	% Get HP value
%         end
%     end
    
    % Cross Validation
    
    CVout = cross_valid(DATAtr,HP_probe,class_train,class_test,CVp);
    
    % Define new optimum HP
    
    if (turn == 1)
        HPoptm = HP_probe;
        min_measure = CVout.measure;
    else
        if (CVout.measure < min_measure)
            HPoptm = HP_probe;
            min_measure = CVout.measure;
        end
    end
    
end

%% END