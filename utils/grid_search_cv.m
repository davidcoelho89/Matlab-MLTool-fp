function [HPoptm] = grid_search_cv(DATA,HPgs,f_train,f_class,CVp)

% --- Optm hyperparameters definition by Grid Search and Cross Validation ---
%
%   [HPoptm] = grid_search_cv1(DATA,HP_gs,f_train,f_class,CVp)
%
%   Input:
%       DATA.
%           input = training attributes                            [p x N]
%           output = training labels                               [Nc x N]
%       HPgs = hyperparameters for grid search                     [struct]
%               (vectors containing values that will be tested)
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's class function       
%       CVp.
%           fold = number of data partitions for cross validation	[cte]
%           type = type of cross validation                         [cte]
%               1: takes into account just accurary
%               2: takes into account also the dicitionary size
%           lambda = trade-off between error and dictionary size   	[cte]
%   Output:
%       HPoptm = Optimum hyperparameters of classifier for dataset [struct]

%% INITIALIZATIONS

% Get General Characteristics of Problem

HP_names = fieldnames(HPgs);	% Names of HP
N_HP = numel(HP_names);         % Number of HP

% Init Auxiliary HyperParameters

for i = 1:N_HP
    HP_name = HP_names{i};              % Get HP name
    HP_values = HPgs.(HP_name);         % Get HP values vector
    HP_aux.(HP_name) = HP_values(1);    % Init auxiliary HP value
end

% Init Auxiliary Variables

index_HP = ones(N_HP,1);        % Index for each HP in grid search
still_searching = 1;            % Signalize end of grid search
turn = 0;                       % number of turns of grid search

%% ALGORITHM

while 1
    
    % Update Turn of Grid Search

    turn = turn + 1;
    % display(turn)
    
    % Cross Validation
    
    if (nargin == 4)
        CVout = cross_valid(DATA,HP_aux,f_train,f_class);
    else
        CVout = cross_valid(DATA,HP_aux,f_train,f_class,CVp);
    end

    % Define new optimum HP
    
    if (turn == 1)
        HPoptm = HP_aux;
        min_metric = CVout.metric;
    else
        if (CVout.metric < min_metric)
            HPoptm = HP_aux;
            min_metric = CVout.metric;
        end
    end

    % Update indexes of HP (uses "digital clock logic")

    i = 1;
    while i <= N_HP
        index_HP(i) = index_HP(i)+1;
        
        HP_name = HP_names{i};
        if index_HP(i) > length(HPgs.(HP_name))
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

    if still_searching == 0
        break;
    end

    % update auxiliary HP

    for j = 1:N_HP
        HP_name = HP_names{j};                      % get HP name
        HP_values = HPgs.(HP_name);                 % get HP values vector
        HP_aux.(HP_name) = HP_values(index_HP(j));  % get HP value
    end

end % end of while

%% END