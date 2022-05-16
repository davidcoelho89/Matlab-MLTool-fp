%% MACHINE LEARNING TOOLBOX

% Classification Algorithms (OOP Based) - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/04/11

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS AND OBJECTS

number_of_realizations = 5;
classifier_name = 'lms';
dataset_name = 'iris';
label_encoding = 'bipolar';
normalization = 'zscore';
split_method = 'random';
percentage_for_training = 0.7;

hp_optm_method = 'random';
hp_optm_max_interations = 10;
hp_optm_cost_function = 'error';
hp_optm_weighting_factor = 0.5;
hp_optm_struct = [];

data_acc = cell(number_of_realizations,1);
classifier_acc = cell(number_of_realizations,1);
stats_tr_acc = cell(number_of_realizations,1);
stats_ts_acc = cell(number_of_realizations,1);

normalizer = dataNormalizer();
normalizer.normalization = normalization;

statsGen1turn = classificationStatistics1turn();
classification_stats_tr = classificationStatisticsNturns();
classification_stats_ts = classificationStatisticsNturns();

%% LOAD CLASSIFIER AND CHOOSE ITS HYPERPARAMETERS

classifier = initializeClassifier(classifier_name);

% % LMS Hyperparameters
classifier.number_of_epochs = 200;
classifier.learning_step = 0.05;
classifier.video_enabled = 0;
classifier.add_bias = 1;

% OLS Hyperparameters
% classifier.approximation = 'pinv';
% classifier.regularization = 0.0001;
% classifier.add_bias = 1;

%% LOAD DATASET

dataset = loadClassificationDataset(dataset_name);
dataset = encodeLabels(dataset,label_encoding);

%% RUN EXPERIMENT

for r = 1:number_of_realizations
    
    disp('Turn and Time');
    disp(r);
    display(datestr(now));
    
    datasets = splitDataset(dataset,split_method,percentage_for_training);
    
    normalizer = normalizer.fit(datasets.data_tr.input);
    datasets.data_tr.input = normalizer.transform(datasets.data_tr.input);
    datasets.data_ts.input = normalizer.transform(datasets.data_ts.input);

    data_acc{r} = datasets;
    
    classifier = classifier.fit(datasets.data_tr.input, ...
                                datasets.data_tr.output);
    
    % For debug:
    if(isprop(classifier,'MQE'))
        figure; plot(1:classifier.number_of_epochs,classifier.MQE,'b-');
    end
    
    classifier_acc{r} = classifier;
    
    Yh_tr = classifier.predict(datasets.data_tr.input);
    Yh_ts = classifier.predict(datasets.data_ts.input);
    
    stats_tr = statsGen1turn.calculate_all(datasets.data_tr.output,Yh_tr);
    stats_ts = statsGen1turn.calculate_all(datasets.data_ts.output,Yh_ts);
    
    stats_tr_acc{r} = stats_tr;
    stats_ts_acc{r} = stats_ts;
    
    classification_stats_tr = classification_stats_tr.addResult(stats_tr);
    classification_stats_ts = classification_stats_ts.addResult(stats_ts);
end

classification_stats_tr = classification_stats_tr.calculate();
classification_stats_ts = classification_stats_ts.calculate();

%% SHOW RESULTS



%% END
































