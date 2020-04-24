function [HP_o] = grid_search_cv2(DATA,HP_cv,f_train,f_class,CVp)

% --- Optm hyperparameters definition by Grid Search and Cross Validation ---
%
%   [HP_o] = grid_search_cv2(DATA,HP_cv,f_train,f_class,CVp)
%
%   Input:
%       DATA.
%           input = training attributes                             [p x N]
%           output = training labels                                [Nc x N]
%       HP_gs = hyperparameters for grid search of classifier
%                (vectors containing values that will be tested)
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%       CVp.
%           fold = number of folds for cross validation             [cte]
%           lambda = trade-off between error and dictionary size   	[cte]
%   Output:
%       HP_o = optimum hyperparameters of classifier for data set

%% INIT

% Get Hyperparameters of Grid Search

if (nargin == 4),
    CVp.lambda = 0.5;
end

% trade-off between error and dictionary size
lambda = CVp.lambda;

% Get General Characteristics of Problem

HP_names = fieldnames(HP_cv);	% Names of HP
N_HP = numel(HP_names);         % Number of HP

% Init optimum and auxiliary hyperparameters

for i = 1:N_HP,
    HP_name = HP_names{i};              % name of HyperParameter
    HP_values = HP_cv.(HP_name);    	% get HP vector of values
    HP_aux.(HP_name) = HP_values(1);	% init auxiliary HP
    HP_o.(HP_name) = HP_values(1);      % init optimum HP
end

index_HP = ones(N_HP,1);	% Index for each HP in grid search
still_searching = 1;     	% Signalize end of grid search
min_metric = 2;             % minimum metric of an HP set (max value = 2)

%% ALGORITHM

while 1,

    % Cross Validation

    CVout = cross_valid2(DATA,HP_aux,f_train,f_class,CVp);
    cv_metric = CVout.Ds + lambda * CVout.err;

    % Define new optimum HP

    if (cv_metric < min_metric),
        HP_o = HP_aux;
        min_metric = cv_metric;
    end

    % Update indexes of HP

    i = 1;
    while i <= N_HP,
        
        index_HP(i) = index_HP(i)+1;

        if index_HP(i) > length(HP_cv.(HP_names{i})),
            if i == N_HP
                still_searching = 0;
            end
            index_HP(i) = 1;
            i = i + 1;
        else
            break;
        end
        
    end

    % if all HP sets were tested, finish the grid search

    if still_searching == 0,
        break;
    end

    % update auxiliary HP

    for j = 1:N_HP,
        HP_name = HP_names{j};                      % get HP name
        HP_values = HP_cv.(HP_name);                % get HP values vector
        HP_aux.(HP_name) = HP_values(index_HP(j));  % get HP value
    end

end % end of while

%% END