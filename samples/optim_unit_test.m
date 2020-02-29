%% Machine Learning ToolBox

% Optimization Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2020/02/17

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 02;              % Which problem will be solved / used
OPT.Nr = 10;              	% Number of repetitions of the algorithm

%% CHOOSE ALGORITHM

% Handler for optimization function

opt_name = 'PSO';
opt_alg = @pso_optm;

%% CHOOSE HYPERPARAMETERS

HP.Ng = 200;        % Max number of generations
HP.Ni = 10;         % Number of subjects (individuals)
HP.W = 0.5;       	% Inertia Factor [0.4 - 0.9]
HP.c1 = 2.05;       % importance of the best local value
HP.c2 = 2.05;       % importance of the best general value
                    %c1 + c2 ~ 4 (for a good aproximation)

%% PROBLEM LOADING

PROB = data_optm_loading(OPT);

%% RUN OPTIMIZATION ALGORITHM

PAR = opt_alg(PROB,HP);

%% RESULTS

figure;
plot(PAR.fit_best);
title('Best fitness')

figure
plot(PAR.fit_mean);
title('Mean Fitness')

%% END