function [HP_o] = grid_search_cv1(DATA,HP_gs,f_train,f_class,CVp)

% --- Optm hyperparameters definition by Grid Search and Cross Validation ---
%
%   [HP_o] = grid_search_cv1(DATA,HP_gs,f_train,f_class,CVp)
%
%   Input:
%       DATA.
%           input = training attributes                             [p x N]
%           output = training labels                                [Nc x N]
%       HP_gs = hyperparameters for grid search                     [struct]
%             (vectors containing values that will be tested)
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's class function       
%       CVp.
%           Nfold = number of data partitions for cross validation  [cte]
%   Output:
%       HP_o = optimum hyperparameters of classifier for data set   [struct]

%% INIT

% Get Hyperparameters of Grid Search

if (nargin == 4),
    CVp.Nfold = 0.5;
end

% Get General Characteristics of Problem

HP_names = fieldnames(HP_gs);	% Names of HP
N_HP = numel(HP_names);         % Number of HP

% Init Optimum and Auxiliary HyperParameters

for i = 1:N_HP,
    HP_name = HP_names{i};              % Get HP name
    HP_values = HP_gs.(HP_name);        % Get HP values vector
    HP_aux.(HP_name) = HP_values(1);    % Init auxiliary HP
    HP_o.(HP_name) = HP_values(1);      % Init optimum HP
end

index_HP = ones(N_HP,1);        % Index for each HP in grid search
end_flag = 0;                   % Signalize end of grid search
max_accuracy = 0;              	% maximum accuracy of an HP set

%% ALGORITHM

while 1,

    % Cross Validation

    accuracy = cross_valid1(DATA,HP_aux,f_train,f_class,CVp);

    % Define new optimum HP

    if (accuracy > max_accuracy),
        HP_o = HP_aux;
        max_accuracy = accuracy;
    end

    % Update indexes of HP

    i = 1;
    while i <= N_HP,
        index_HP(i) = index_HP(i)+1;
        
        HP_name = HP_names{i};
        if index_HP(i) > length(HP_gs.(HP_name)),
            if i == N_HP
                end_flag = 1;
            end
            index_HP(i) = 1;
            i = i + 1;
        else
            break;
        end
    end

    % if all HP sets were tested, finish the grid search

    if end_flag == 1,
        break;
    end

    % update auxiliary HP

    for j = 1:N_HP,
        HP_name = HP_names{j};                     % get HP name
        HP_values = HP_gs.(HP_name);               % get HP values vector
        HP_aux.(HP_name) = HP_values(index_HP(j)); % get HP value
    end

end % end of while

%% END