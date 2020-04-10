function [HP_o] = cross_valid_gs2(DATA,CVp,HP_cv,f_train,f_class)

% --- Optimum hyperparameters definition by Cross Validation and Grid Search ---
%
%   [HP_o] = cross_valid_gs2(DATA, CVp, HP_cv, f_train, f_class)
%
%   Input:
%       DATA.
%           input = training attributes                             [p x N]
%           output = training labels                                [Nc x N]
%       CVp.
%           fold = number of folds for cross validation             [cte]
%           lambda = trade-off between error and dictionary size   	[cte]
%       HP_cv = hyperparameters for cross validation of classifier
%                (vectors containing values that will be tested)
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%   Output:
%       HP_o = optimum hyperparameters of classifier for data set

%% INIT

HP_names = fieldnames(HP_cv);	% Names of HP
N_HP = numel(HP_names);         % Number of HP

index_HP = ones(N_HP,1);        % Index for each HP in grid search

% Init optimum and auxiliary hyperparameters
for j = 1:N_HP,
    hp = HP_cv.(HP_names{j});     	% get HP vector of values
    HP_aux.(HP_names{j}) = hp(1);   % auxiliary HP
    HP_o.(HP_names{j}) = hp(1);     % optimum HP
end

end_flag = 0;           % Signalize end of grid search
min_metric = 2;         % minimum metric of an HP set (max value = 2)
lambda = CVp.lambda;	% trade-off between error and dictionary size

%% ALGORITHM

while 1,

    % Cross Validation

    CVout = cross_valid2(DATA,HP_aux,CVp,f_train,f_class);
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
        hp = HP_cv.(HP_names{j});               % get HP vector
        HP_aux.(HP_names{j}) = hp(index_HP(j)); % get HP value
    end

end % end of while

%% END