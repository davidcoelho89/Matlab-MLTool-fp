function [HP_o] = cross_valid_gs3(DATA,CVp,HP_cv,f_train,f_class)

% --- Optimum hyperparameters definition by Cross Validation and Grid Search ---
%
%   [HP_o] = cross_valid_gs3(DATA, CVp, HP_cv, f_train, f_class)
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

HP_o = DATA+CVp+HP_cv+f_train+f_class;

%% ALGORITHM


%% END