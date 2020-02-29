function [PAR] = pso_optm(PROB,HP)

% --- Particle Swarm Optimization ---
%
%   [PAR] = pso_optm(PROB,HP)
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
%           Ng = Max number of generations                	[cte]
%           Ni = Number of subjects                       	[cte]
%           W = Inertia Factor                              [0.4 - 0.9]
%           c1 = importance of the best local value         [cte]
%           c2 = importance of the best general value       [cte]
%   Output:
%       PAR.
%           "names": depends on the problem to be solved

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    HPaux.Ng = 200;    	% Max number of generations
    HPaux.Ni = 10;     	% Number of subjects
    HPaux.W = 0.5;  	% Inertia Factor [0.4 - 0.9]
    HPaux.c1 = 2.05;  	% importance of the best local value
    HPaux.c2 = 2.05;  	% importance of the best general value
                        %c1 + c2 ~ 4 (for a good aproximation)
	HP = HPaux;
else
    if (~(isfield(HP,'Ng'))),
        HP.Ng = 400;
    end
    if (~(isfield(HP,'Ni'))),
        HP.Ni = 10;
    end
    if (~(isfield(HP,'W'))),
        HP.W = 0.5;
    end
    if (~(isfield(HP,'c1'))),
        HP.c1 = 2.05;
    end
    if (~(isfield(HP,'c2'))),
        HP.c2 = 2.05;
    end
end

%% INITIALIZATIONS

% Get Hyperparameters

Ng = HP.Ng;                     % Max number of generations
Ni = HP.Ni;                     % Number of subjects (individuals)
c1 = HP.c1;                     % importance of the best local value
c2 = HP.c2;                     % importance of the best general value
W = HP.W;                       % Inertia Factor [0.4 - 0.9]

% Get problem characteristics

fitness = PROB.fitness;         % Fitness function of each subject
minimum = PROB.min;             % Minimum value of variables
maximum = PROB.max;             % Maximum value of variables
names = PROB.names;             % Variables names
Nc = length(names);             % No of variables to be optimized
maximization = PROB.maximization;

% Problem Initialization

X = zeros(Nc,Ni);               % Initial population
for j = 1:Nc,
    X(j,:) = minimum(j) + (maximum(j) - minimum(j))*rand(1,Ni);
end

V   = zeros(Nc,Ni);             % Initial speed

r1 = rand(Nc,Ni);               % aleatory value from 0 to 1
r2 = rand(Nc,Ni);               % aleatory value from 0 to 1

fit_best = zeros(1,Ng);         % Best Values of fitness function
fit_mean = zeros(1,Ng);         % Mean Values of fitness functions

FIT = fitness(X);               % Initial fitness of population

pbest = [X ; FIT.F];            % Initial best local positions

if(maximization == 1),
    [best_val,ind] = max(FIT.F);
else
    [best_val,ind] = min(FIT.F);
end

gbest = [X(:,ind) ; best_val];  % Initial best global positions

%% ALGORITHM

% Apply optimization algorithm

for t = 1:Ng,
    V = pso_vel(X,V,pbest,gbest,W,c1,c2,r1,r2);     % New velocity
    X = pso_pos(X,V);                               % New positions
    FIT = fitness(X);                               % New fitness
    [pbest,gbest] = pso_best(pbest,gbest,X,FIT.F,maximization);
    if(maximization == 1),
        fit_best(t) = max(FIT.F);                   % hold max fitness
    else
        fit_best(t) = min(FIT.F);                   % hold min fitness
    end
    fit_mean(t) = mean(FIT.F);                      % hold mean fitness
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