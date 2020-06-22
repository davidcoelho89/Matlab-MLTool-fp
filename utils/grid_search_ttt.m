function [HPoptm] = grid_search_ttt(DATA,HPgs,f_train,f_class,PSp)

% --- Optm hyperparameters definition by Grid Search for Sequential Learn ---
%
%   [HPoptm] = grid_search_ttt(DATA,HPgs,f_train,f_class,PSp)
%
%   Input:
%       DATA.
%           input = training attributes                            [p x N]
%           output = training labels                               [Nc x N]
%       HPgs = hyperparameters for grid searh of classifier
%             (vectors containing values that will be tested)
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%       PSp.
%           iterations = number of times the data is 
%                        presented to the algorithm
%           type = type of cross validation                         [cte]
%               1: takes into account just accurary
%               2: takes into account also the dicitionary size
%           lambda = trade-off between error and dictionary size    [0 - 1]
%   Output:
%       HPoptm = optimum hyperparameters of classifier for data set

%% INITIALIZATIONS

% Get General Characteristics of Problem

HyperParameterNames = fieldnames(HPgs);
NumberOfHyperParameters = numel(HyperParameterNames);

% Init Auxiliary HyperParameters

for i = 1:NumberOfHyperParameters,
    HyperParameterName = HyperParameterNames{i};
    HpValuesVector = HPgs.(HyperParameterName);
    HPaux.(HyperParameterName) = HpValuesVector(1);
end

% Init Auxiliary Variables

IndexOfHyperParameters = ones(NumberOfHyperParameters,1);
still_searching = 1;        % Signalize end of grid search
turn = 0;                 	% number of turns of grid search

%% ALGORITHM

while 1,
    
    % Update Turn of Grid Search
    
    turn = turn + 1;

    % "Interleaved Test-Then-Train" or "Prequential" Method
    
    if (nargin == 4),
        PSout = presequential_valid(DATA,HPaux,f_train,f_class);
    else
        PSout = presequential_valid(DATA,HPaux,f_train,f_class,PSp);
    end

    % Define New Optimum HyperParameters
    
    if (turn == 1),
        HPoptm = PSout.PAR;
        min_measure = PSout.measure;
    else
        if (PSout.measure < min_measure),
            HPoptm = PSout.PAR;
            min_measure = PSout.measure;
        end
    end

    % Update indexes of HP (uses "digital clock logic")
    
    i = 1;
    while i <= NumberOfHyperParameters,
        
        IndexOfHyperParameters(i) = IndexOfHyperParameters(i) + 1;
        number_of_values = length(HPgs.(HyperParameterNames{i}));
        if (IndexOfHyperParameters(i) > number_of_values)
            if i == NumberOfHyperParameters,
                still_searching = 0;
            end
            IndexOfHyperParameters(i) = 1;
            i = i + 1;
        else 
            break;
        end
        
    end
    
    % If all HP sets were tested, finish the grid search
    
    if still_searching == 0,
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