%% MACHINE LEARNING TOOLBOX

% Classification Algorithms (OOP Based) - Unit Test
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
model_name = 'knn';
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

classifier = initializeClassifier(model_name);

if(strcmp(model_name,'ols'))
    classifier.approximation = 'pinv';
    classifier.regularization = 0.0001;
    classifier.add_bias = 1;

elseif(strcmp(model_name,'lms'))
    classifier.number_of_epochs = 200;
    classifier.learning_step = 0.05;
    classifier.video_enabled = 0;
    classifier.add_bias = 1;

elseif(strcmp(model_name,'knn'))
    classifier.distance_measure = 2;
    classifier.nearest_neighbors = 1;
    classifier.knn_aproximation = 'majority_voting'; % 'weighted_knn'
    classifier.kernel_type = 'none';

elseif(strcmp(model_name,'spok'))
    classifier.number_of_epochs = 1;
    classifier.is_stationary = 0;
    classifier.design_method = 'one_dicitionary_per_class';
    classifier.sparsification_strategy = 'ald';
    classifier.v1 = 0.1;
    classifier.v2 = 0.9;
    classifier.update_strategy = 'wta';
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
end

%% DATA LOADING AND PRE-PROCESSING

dataset = loadClassificationDataset(dataset_name);
dataset = encodeLabels(dataset,label_encoding);

%% RUN EXPERIMENT

disp('Begin Algorithm');

for realization = 1:number_of_realizations
    
% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(realization);
display(datestr(now));

% %%%%%%%%%%%% SPLIT AND NORMALIZE DATASETS %%%%%%%%%%%%%%

datasets = splitDataset(dataset,split_method,percentage_for_training);

normalizer = normalizer.fit(datasets.data_tr.input);
datasets.data_tr.input = normalizer.transform(datasets.data_tr.input);
datasets.data_ts.input = normalizer.transform(datasets.data_ts.input);

data_acc{realization} = datasets;

% %%%%%%%%%%%% HYPERPARAMETER OPTIMIZATION %%%%%%%%%%%%%%%

% ToDo - All

% %%%%%%%%%%%%%%%% SYSTEM'S ESTIMATION %%%%%%%%%%%%%%%%%%%

classifier = classifier.fit(datasets.data_tr.input, ...
                            datasets.data_tr.output);

classifier_acc{realization} = classifier;

% %%%%%%%% SYSTEM'S PREDICTION AND STATISTICS %%%%%%%%%%%%

classifier = classifier.predict(datasets.data_tr.input);
stats_tr = statsGen1turn.calculate_all(datasets.data_tr.output,classifier.Yh);

classifier = classifier.predict(datasets.data_ts.input);
stats_ts = statsGen1turn.calculate_all(datasets.data_ts.output,classifier.Yh);

classification_stats_tr = classification_stats_tr.addResult(stats_tr);
classification_stats_ts = classification_stats_ts.addResult(stats_ts);

end

%% RESULTS / STATISTICS - CALCULATE

classification_stats_tr = classification_stats_tr.calculate_all();
classification_stats_ts = classification_stats_ts.calculate_all();

%% RESULTS / STATISTICS - SHOW

cell_stats_compare{1,1} = classification_stats_tr;
cell_stats_compare{1,2} = classification_stats_ts;

cell_stats_names{1,1} = 'Train';
cell_stats_names{1,2} = 'Test';

showAccuracyComparison(cell_stats_compare,cell_stats_names);
showErrorComparison(cell_stats_compare,cell_stats_names);

disp(classification_stats_ts.acc_vect');

%% END