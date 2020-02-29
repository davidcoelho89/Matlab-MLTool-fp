function [PAR] = de_optm(PROB,HP)

% --- Differential Evolution for Optimization ---
%
%   [PAR] = de_optm(PROB,HP)
%
%   Input:
%       PROB: Problem to be solved
%           names = Variables names                         [cell p x 1]
%           values = Possible values for variables          [cell p x 1]
%           max = Maximum value of variables                [vect p x 1]
%           min = Minimum value of variables                [vect p x 1]
%           types = Data type (int or float)                [cell p x 1]
%           fitness = Handle for fitness function           [handler]
%           maximization = type of problem -> max ou min  	[0 or 1]
%       HP: Hyperparameters
%           Ng = Max number of generations                 	[cte]
%           Ni = Number of subjects                        	[cte]
%           Pc = Crossing probability                    	[cte]
%           B = Difference amplification                    [cte]
%           Ss = Selection Strategy                       	[cte]
%               1: Tournament
%           El = Elitism                                    [cte]
%   Output:
%       PAR.
%           "names": depends on the problem to be solved

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    HPaux.Ng = 200;    	% Max number of generations
    HPaux.Ni = 10;     	% Number of subjects
    HPaux.Pc = 0.8;   	% Crossing probability
    HPaux.B = 0.5;    	% Difference anplification
    HPaux.Ss = 1;     	% Selection Srategy
    HPaux.El = 1;      	% Elitism
	HP = HPaux;
else
    if (~(isfield(HP,'Ng'))),
        HP.Ng = 200;
    end
    if (~(isfield(HP,'Ni'))),
        HP.Ni = 10;
    end
    if (~(isfield(HP,'Pc'))),
        HP.Pc = 0.8;
    end
    if (~(isfield(HP,'B'))),
        HP.B = 0.5;
    end
    if (~(isfield(HP,'Ss'))),
        HP.Ss = 1;
    end
    if (~(isfield(HP,'El'))),
        HP.El = 1;
    end
end

%% INIT

% Get Hyperparameters

Ng = HP.Ng;           	% Max number of generations
Ni = HP.Ni;           	% Number of subjects
Pc = HP.Pc;           	% Crossing probability 
B = HP.B;               % Difference anplification

% Get problem characteristics

fitness = PROB.fitness;	% Fitness function of each subject
minimum = PROB.min;     % Minimum value of variables
maximum = PROB.max;     % Maximum value of variables
names = PROB.names;    	% Variables names
Nc = length(names);     % No of chromossomes 
                        % (each chromossome is a variable to be optimized)

% Problem Initialization

X = zeros(Nc,Ni);             % Initial population
for j = 1:Nc,
    X(j,:) = minimum(j) + (maximum(j) - minimum(j))*rand(1,Ni);
end
fit_best = zeros(1,Ng); 	% Best Value of fitness function
fit_mean = zeros(1,Ng);     % Mean Value of fitness functions

%% ALGORITHM

% Apply optimization algorithm

for t = 1:Ng,
    FITx = fitness(X);                  % calculate fitness of population
    if(PROB.maximization == 1),
        fit_best(t) = max(FITx.F);      % hold max fitness
    else
        fit_best(t) = min(FITx.F);      % hold min fitness
    end
    fit_mean(t) = sum(FITx.F)/Ni;       % hold mean fitness
    U = de_mutate(X,B);                 % mutate individuals
    U = de_crossing(X,U,Pc);            % cross selected individuals
    FITu = fitness(U);                  % calculate fitness of trial vectors
    X = de_select(X,U,FITx.F,FITu.F);   % build next generation
end

% Get best individual and variables

FIT = fitness(X);
[~,best] = min(FIT.F);
for i = 1:Nc,
    PAR.(names{i}) = FIT.X(i,best);
end

%% FILL OUTPUT STRUCTURE

PAR.X = X;
PAR.FIT = FIT;
PAR.fit_best = fit_best;
PAR.fit_mean = fit_mean;

%% END