function [PAR] = ga_optm(PROB,HP)

% --- Genetic Algorithm for Optimization ---
%
%   [PAR] = ga_optm(HP,PROB)
%
%   Input:
%       PROB: Problem to be solved
%           names = Variables names                         [cell p x 1]
%           values = Possible values for variables          [cell p x 1]
%           max = Maximum value of variables                [vect p x 1]
%           min = Minimum value of variables                [vect p x 1]
%           types = Data type (int or float)                [cell p x 1]
%           fitness = Handle for fitness function           [handler]
%           maximization = type of problem -> max ou min   	[0 or 1]
%       HP: Hyperparameters
%           Ng = Max number of generations                      [cte]
%           Ni = Number of subjects                             [cte]
%           Cl = Chromossomes length (no of genes / resolution) [cte]
%           Pc = Crossing probability                           [cte]
%           Pm = Mutation probability                           [cte]
%           Ss = Selection Strategy                             [cte]
%               1: Tournament
%           El = Elitism                                        [cte]
%   Output:
%       PAR.
%           "names": depends on the problem to be solved

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    HPaux.Ng = 200;    	% Max number of generations
    HPaux.Ni = 10;     	% Number of subjects
    HPaux.Cl = 10;     	% Chromossomes length
    HPaux.Pc = 0.8;    	% Crossing probability 
    HPaux.Pm = 0.01;    % Mutation probability
    HPaux.Ss = 1;       % Tournament Selection
    HPaux.El = 1;       % Presence of Elitism
	HP = HPaux;
else
    if (~(isfield(HP,'Ng'))),
        HP.Ng = 400;
    end
    if (~(isfield(HP,'Ni'))),
        HP.Ni = 10;
    end
    if (~(isfield(HP,'Cl'))),
        HP.Cl = 10;
    end
    if (~(isfield(HP,'Pc'))),
        HP.Pc = 0.8;
    end
    if (~(isfield(HP,'Pm'))),
        HP.Pm = 0.01;
    end
    if (~(isfield(HP,'Ss'))),
        HP.Ss = 1;
    end
    if (~(isfield(HP,'El'))),
        HP.El = 1;
    end
end

%% INITIALIZATIONS

% Get Hyperparameters

Ng = HP.Ng;           	% Max number of generations
Ni = HP.Ni;           	% Number of subjects
Cl = HP.Cl;            	% Chromossomes length
Pc = HP.Pc;           	% Crossing probability 
Pm = HP.Pm;           	% Mutation probability
Ss = HP.Ss;             % Selection Strategy
El = HP.El;            	% Elitism

% Get problem characteristics

fitness = PROB.fitness;	% Fitness function of each subject
names = PROB.names;    	% Variables names
Nc = length(names);     % No of chromossomes 
                        % (each chromossome is a variable to be optimized)

% Problem Initialization

P = round(rand(Nc*Cl,Ni));  % Initial population
fit_best = zeros(1,Ng); 	% Best Value of fitness function
fit_mean = zeros(1,Ng);     % Mean Value of fitness functions

%% ALGORITHM

% Apply optimization algorithm

for t = 1:Ng,
    FIT = fitness(P);                       % calculate fitness
    if(PROB.maximization == 1),
        fit_best(t) = max(FIT.F);           % hold max fitness
    else
        fit_best(t) = min(FIT.F);        	% hold min fitness
    end
    fit_mean(t) = mean(FIT.F);              % hold mean fitness
    S = ga_select(P,FIT.Fn,Ss);             % select individuals for crossing
    P = ga_crossing(P,S,Pc,FIT.Fn,Nc,El);   % cross selected individuals
    P = ga_mutate(P,Pm,El);                 % mutate individuals
end

% Get best individual and variables

FIT = fitness(P);
[~,best] = min(FIT.F);
for i = 1:Nc,
    PAR.(names{i}) = FIT.X(i,best);
end

%% FILL OUTPUT STRUCTURE

PAR.P = P;
PAR.FIT = FIT;
PAR.fit_best = fit_best;
PAR.fit_mean = fit_mean;

%% END