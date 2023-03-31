%% MACHINE LEARNING TOOLBOX

% Clustering Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2023/03/20

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

number_of_realizations = 10;
dataset_name = 'iris';
model_name = 'wta';
normalization = 'zscore';

normalizer = dataNormalizer();
normalizer.normalization = normalization;

%% LOAD CLUSTERING MODEL AND CHOOSE ITS HYPERPARAMETERS

clusteringModel = initializeClusteringModel(model_name);

%% CHOOSE HYPERPARAMETERS TO BE OPTIMIZED



%% ACCUMULATORS



%% HANDLERS FOR CLASSIFICATION FUNCTIONS



%% DATA LOADING, PRE-PROCESSING, VISUALIZATION



%% HOLD OUT / NORMALIZE / SHUFFLE / HPO / TRAINING / TEST / STATISTICS



%% RESULTS / STATISTICS



%% GRAPHICS - OF LAST TURN



%% SAVE VARIABLES AND VIDEO



%% END