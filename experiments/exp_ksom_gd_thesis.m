%% Machine Learning ToolBox

% KSOM-GD Comparison Tests
% Author: David Nascimento Coelho
% Last Update: 2023/03/02

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 05;        % Number of repetitions of each algorithm
OPT.alg = 'ksom';   % Which Classifier will be used
OPT.prob = 06;      % Which problem will be solved / used
OPT.prob2 = 01;     % When it needs an specification of data set
OPT.norm = 0;       % Normalization definition
OPT.lbl = 1;        % Data labeling definition
OPT.hold = 02;      % Hold out method
OPT.ptrn = 0.7;     % Percentage of samples for training
OPT.hpo = 'none';   % 'grid' ; 'random' ; 'none'

OPT.savefile = 0;   % decides if file will be saved

OPT.calculate_bin = 0;  % decides to calculate binary statistics
OPT.class_1_vect = 1;   % [2,3] which classes belongs together

% Prototypes' labeling definition

prot_lbl = 1;               % = 1 / 2 / 3

% Metaparameters

MP.max_it = 100;   	% Maximum number of iterations (random search)
MP.fold = 5;     	% number of data partitions (cross validation)
MP.cost = 2;        % Takes into account also the dicitionary size
MP.lambda = 2.0;    % Jpbc = Ds + lambda * Err