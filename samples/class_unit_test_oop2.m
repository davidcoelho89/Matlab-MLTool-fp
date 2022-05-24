%% MACHINE LEARNING TOOLBOX

% Classification Algorithms (OOP Based) - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/04/11

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

experiment = classificationExperiment();

% experiment.number_of_realizations = 5;
% experiment.classifier_name = 'lms';
% experiment.dataset_name = 'iris';
% experiment.label_encoding = 'bipolar';
% experiment.normalization = 'zscore';
% experiment.split_method = 'random';
% experiment.percentage_for_training = 0.7;
% experiment.filename = 'filex.mat';
% experiment.hp_optm_method = 'random';
% experiment.hp_optm_max_interations = 10;
% experiment.hp_optm_cost_function = 'error';
% experiment.hp_optm_weighting_factor = 0.5;
% experiment.hp_optm_struct = [];

%% LOAD CLASSIFIER AND CHOOSE ITS HYPERPARAMETERS

% experiment.classifier = lmsClassifier();
% 
% experiment.classifier.number_of_epochs = 200;
% experiment.classifier.learning_step = 0.05;
% experiment.classifier.video_enabled = 0;
% experiment.classifier.add_bias = 1;

%% LOAD DATASET

% experiment.dataset_name = 'dermatology';
% experiment.dataset = loadClassificationDataset(experiment.dataset_name);

%% RUN EXPERIMENT

experiment = experiment.run();

%% SHOW RESULTS

experiment.show_results();

%% END
































