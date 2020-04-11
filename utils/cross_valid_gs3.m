function [HPoptimum] = cross_valid_gs3(DATA,HPgs,f_train,f_class,GSp)

% --- Optimum hyperparameters definition by Cross Validation and Grid Search ---
%
%   [HP_o] = cross_valid_gs3(DATA,HPgs,f_train,f_class,GSp)
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
    lambda = 0.5;
else
    lambda = GSp.lambda;	% trade-off between error and dictionary size
end

% Get General Characteristics of Problem

HyperParameterNames = fieldnames(HPgs);
NumberOfHyperParameters = numel(HyperParameterNames);
IndexOfHyperParameters = ones(NumberOfHyperParameters,1);

% Init Optimum and Auxiliary HyperParameters

for hp = 1:NumberOfHyperParameters,
    HyperParameterName = HyperParameterNames{hp};
    HpValuesVector = HPgs.(HyperParameterName);
    HPauxiliary.(HyperParameterName) = HpValuesVector(1);
    HPoptimum.(HyperParameterName) = HpValuesVector(1);
end

% Init Auxiliary Variables

end_flag = 0;           % Signalize end of grid search
min_metric = 2;         % minimum metric of an HP set (max value = 2)

%% ALGORITHM

while 1,

    % "Interleaved Test-Then-Train" or "Prequential"
    
    PresequentialOut = presequential(DATA,HPauxiliary,f_train,f_class);
    cv_metric = PresequentialOut.Ds + lambda * PresequentialOut.err;

    % Define new optimum HP

    if (cv_metric < min_metric),
        HPoptimum = HPauxiliary;
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
         HPauxiliary.(HyperParameterName) = ...
                        HpValuesVector(IndexOfHyperParameters(i));
     end

end

%% END