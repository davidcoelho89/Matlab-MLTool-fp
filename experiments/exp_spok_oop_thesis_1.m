%% Machine Learning ToolBox

% Spok Algorithm (OOP Based)
% Author: David Nascimento Coelho
% Last Update: 2022/12/22

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS AND OBJECTS

number_of_realizations = 10;
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

data_acc = cell(number_of_realizations,1);
classifier_acc = cell(number_of_realizations,1);

normalizer = dataNormalizer();
normalizer.normalization = normalization;

statsGen1turn = classificationStatistics1turn();
classification_stats_tr = classificationStatisticsNturns(number_of_realizations);
classification_stats_ts = classificationStatisticsNturns(number_of_realizations);

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

%% RUN EXPERIMENT

for realization = 1:number_of_realizations

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(realization);
display(datestr(now));

datasets = splitDataset(dataset,split_method,percentage_for_training);

normalizer = normalizer.fit(datasets.data_tr.input);
datasets.data_tr.input = normalizer.transform(datasets.data_tr.input);
datasets.data_ts.input = normalizer.transform(datasets.data_ts.input);

data_acc{realization} = datasets;



end

%% END














