function [PROBout] = data_optm_loading(OPTION)

% --- Selects a Data Base for Optimization ---
%
%   [PROBout] = data_optm_loading(OPTION)
%
%   Input:
%       OPTION.
%	    prob = which data base will be used             [cte]
%           01: Rastrigin
%   Output:
%       PROB.
%           names = variables names                         [cell p x 1]
%           values = possible values for variables          [cell p x 1]
%           max = maximum value of variables                [vect p x 1]
%           min = minimum value of variables                [vect p x 1]
%           types = data type (int or float)                [cell p x 1]
%           fitness = handle for fitness function           [handler]
%           maximization = type of problem -> max ou min   	[0 or 1]

%% INITIALIZATION

PROB = struct('names',[],'max',[],'min',[],'types',[],'fitness',[]);

choice = OPTION.prob;

%% ALGORITHM

switch (choice),
    
    case 1,
        % Rastrigin function binary
        PROB.names = {'x1','x2'};
        PROB.min = [-5.12,-5.12];
        PROB.max = [5.12,5.12];
        PROB.types = {'float','float'};
        PROB.fitness = @rastrigin_fitness_bin;
        PROB.maximization = 0;
    case 2,
    	% Rastrigin function decimal
        PROB.names = {'x1','x2'};
        PROB.min = [-5.12,-5.12];
        PROB.max = [5.12,5.12];
        PROB.types = {'float','float'};
        PROB.fitness = @rastrigin_fitness_dec;
        PROB.maximization = 0;
    otherwise
        % None of the sets
        disp('Unknown Data Base. Void Structure Created')
end

%% FILL OUTPUT STRUCTURE

PROBout = PROB;

%% END