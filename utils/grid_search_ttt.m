function [HPoptm] = grid_search_ttt(DATA,HPgs,f_train,f_class,GSp)

% --- Optm hyperparameters definition by Grid Search for Sequential Learn ---
%
%   [HP_o] = grid_search_ttt(DATA,HPgs,f_train,f_class,GSp)
%
%   Input:
%       DATA.
%           input = training attributes                             [p x N]
%           output = training labels                                [Nc x N]
%       HPgs = hyperparameters for grid searh of classifier
%             (vectors containing values that will be tested)
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%       GSp.
%           lambda = trade-off between error and dictionary size   	[0 - 1]
%   Output:
%       HP_o = optimum hyperparameters of classifier for data set

%% INIT

% Get Hyperparameters

if (nargin == 4),
    GSp.lambda = 0.5;
end

% trade-off between error and dictionary size
lambda = GSp.lambda;

% Get General Characteristics of Problem

HyperParameterNames = fieldnames(HPgs);
NumberOfHyperParameters = numel(HyperParameterNames);

% Init Optimum and Auxiliary HyperParameters

for i = 1:NumberOfHyperParameters,
    HyperParameterName = HyperParameterNames{i};
    HpValuesVector = HPgs.(HyperParameterName);
    HPaux.(HyperParameterName) = HpValuesVector(1);
    HPoptm.(HyperParameterName) = HpValuesVector(1);
end

% Init Auxiliary Variables

IndexOfHyperParameters = ones(NumberOfHyperParameters,1);
end_flag = 0;           % Signalize end of grid search
min_metric = 2;         % minimum metric of an HP set (max value = 2)

%% ALGORITHM

while 1,

    % "Interleaved Test-Then-Train" or "Prequential" Method
    
    PresequentialOut = presequential_valid1(DATA,HPaux,f_train,f_class);
    cv_metric = PresequentialOut.Ds + lambda * PresequentialOut.err;

    % Define new optimum HP

    if (cv_metric < min_metric),
        HPoptm = HPaux;
        min_metric = cv_metric;
    end

    % Update indexes of HP
    
    i = 1;
    while i <= NumberOfHyperParameters,
        
        IndexOfHyperParameters(i) = IndexOfHyperParameters(i) + 1;
        number_of_values = length(HPgs.(HyperParameterNames{i}));
        if (IndexOfHyperParameters(i) > number_of_values)
            if i == NumberOfHyperParameters,
                end_flag = 1;
            end
            IndexOfHyperParameters(i) = 1;
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
     
     for j = 1:NumberOfHyperParameters,
         HyperParameterName = HyperParameterNames{j};
         HpValuesVector = HPgs.(HyperParameterName);
         HPaux.(HyperParameterName) = ...
                        HpValuesVector(IndexOfHyperParameters(j));
     end

end

%% END