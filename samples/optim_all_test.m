%% Machine Learning ToolBox

% Optimization Algorithms - General Tests
% Author: David Nascimento Coelho
% Last Update: 2020/02/17

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT1.prob = 01;  	% Which problem will be solved / used
OPT1.Nr = 10;    	% Number of repetitions of the algorithm

OPT2.prob = 02;  	% Which problem will be solved / used
OPT2.Nr = 10;    	% Number of repetitions of the algorithm

%% CHOOSE ALGORITHMS

% Handler for optimization function

opt_names = {'GA','DE','PSO'};

%% CHOOSE HYPERPARAMETERS

GAhp.Ng = 400;          % Max number of generations
GAhp.Ni = 12;           % Number of subjects (individuals)
GAhp.Cl = 10;           % Chromossomes length (number of genes / resolution)
GAhp.Pc = 0.8;          % Crossing probability
GAhp.Pm = 0.1;          % Mutation probability
GAhp.Ss = 1;            % Selection Srategy
GAhp.El = 1;            % Elitism

DEhp.Ng = 200;          % Max number of generations
DEhp.Ni = 10;           % Number of subjects (individuals)
DEhp.Pc = 0.8;          % Crossing probability
DEhp.B = 0.5;           % Difference amplification
DEhp.Ss = 1;            % Selection Srategy
DEhp.El = 1;            % Elitism

PSOhp.Ng = 200;         % Max number of generations
PSOhp.Ni = 10;          % Number of subjects (individuals)
PSOhp.W = 0.5;       	% Inertia Factor [0.4 - 0.9]
PSOhp.c1 = 2.05;        % importance of the best local value
PSOhp.c2 = 2.05;        % importance of the best general value
                        %c1 + c2 ~ 4 (for a good aproximation)

ACOhp.Ng = 200;         % Max number of generations
ACOhp.Ni = 10;          % Number of subjects (individuals)
                        
%% PROBLEM LOADING

PROB1 = data_optm_loading(OPT1);
PROB2 = data_optm_loading(OPT2);

%% RUN OPTIMIZATION ALGORITHMS

GApar = ga_optm(PROB1,GAhp);
DEpar = de_optm(PROB2,DEhp);
PSOpar = pso_optm(PROB2,PSOhp);

%% RESULTS

figure;
plot(GApar.fit_best);
title('Best fitness for GA algorithm')

figure
plot(GApar.fit_mean);
title('Mean Fitness for GA algorithm')

figure;
plot(DEpar.fit_best);
title('Best fitness for DE algorithm')

figure
plot(DEpar.fit_mean);
title('Mean Fitness for DE algorithm')

figure;
plot(PSOpar.fit_best);
title('Best fitness for PSO algorithm')

figure
plot(PSOpar.fit_mean);
title('Mean Fitness for PSO algorithm')

%% END