%% Machine Learning ToolBox

% Spok Algorithm (OOP Based)
% Author: David Nascimento Coelho
% Last Update: 2022/12/22

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% EXPERIMENT PARAMETERS AND OBJECTS

percentage_for_training = 0.7;
dataset_name = 'iris';
model_name = 'spok';
normalization = 'zscore';
label_encoding = 'bipolar';
split_method = 'random';

hp_optm_method = 'random';
hp_optm_max_interations = 10;
hp_optm_cost_function = 'error';
hp_optm_weighting_factor = 0.5;
hp_optm_struct = [];

normalizer = dataNormalizer();
normalizer.normalization = normalization;

statsGen1turn = classificationStatistics1turn();

%% LOAD CLASSIFIER AND CHOOSE ITS HYPERPARAMETERS

classifier = spokClassifier();

classifier.number_of_epochs = 1;
classifier.is_stationary = 1;
classifier.design_method = 'one_dicitionary_per_class';
classifier.sparsification_strategy = 'ald';
classifier.v1 = 0.1;
classifier.v2 = 0.9;
classifier.update_strategy = 'lms';
classifier.update_rate = 0.1;
classifier.pruning_strategy = 'error_score_based';
classifier.min_score = -10;
classifier.max_prototypes = 600;
classifier.min_prototypes = 2;
classifier.video_enabled = 0;
classifier.nearest_neighbors = 1;
classifier.knn_aproximation = 'majority_voting';
classifier.kernel_type = 'gaussian';
classifier.regularization = 0.001;
classifier.sigma = 2;
classifier.alpha = 1;
classifier.theta = 1;
classifier.gamma = 2;

%% DATA LOADING AND PRE-PROCESSING

dataset = loadClassificationDataset(dataset_name);
dataset = encodeLabels(dataset,label_encoding);

%% HOLD OUT AND NORMALIZATION

datasets = splitDataset(dataset,split_method,percentage_for_training);
data_tr = datasets.data_tr;
data_ts = datasets.data_ts;

normalizer = normalizer.fit(data_tr.input);
data_tr.input = normalizer.transform(data_tr.input);
data_ts.input = normalizer.transform(data_ts.input);

%% TRAIN, TEST, STATISTICS

% classifier = classifier.fit(data_tr.input,data_tr.output);
% 
% classifier = classifier.predict(data_tr.input);
% stats_tr = statsGen1turn.calculate_all(data_tr.output,classifier.Yh);
% 
% classifier = classifier.predict(data_ts.input);
% stats_ts = statsGen1turn.calculate_all(data_ts.output,classifier.Yh);

%% RESULTS / STATISTICS - SHOW




%% END






























