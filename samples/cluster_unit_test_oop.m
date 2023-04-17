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

statsGen1turn = clusteringStatistics1turn();
clusteringStatsNturns = clusteringStatisticsNturns(number_of_realizations);

%% LOAD CLUSTERING MODEL AND CHOOSE ITS HYPERPARAMETERS

clusteringModel = initializeClusteringModel(model_name);

if(strcmp(model_name,'wta'))
    clusteringModel.distance_measure = 2;
    clusteringModel.kernel_type = 'none';
    clusteringModel.number_of_epochs = 20;
    clusteringModel.number_of_prototypes = 20;
    clusteringModel.initialization_type = 'random_samples';
    clusteringModel.learning_type = 4;
    clusteringModel.learning_step_initial = 0.7;
    clusteringModel.learning_step_final = 0.01;
    clusteringModel.video_enabled = 0;
    
end

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

dataset = loadClassificationDataset(dataset_name);

normalizer = normalizer.fit(dataset.input);
dataset.input = normalizer.transform(dataset.input);

%% RUN EXPERIMENT

disp('Begin Algorithm');

for realization = 1:number_of_realizations

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(realization);
display(datestr(now));

% %%%%%%%%%%%%%%%%%% MODEL BUILDING %%%%%%%%%%%%%%%%%%%%%%

clusteringModel = clusteringModel.fit(dataset.input);

% %%%%%%%% MODEL'S PREDICTION AND STATISTICS %%%%%%%%%%%%

clusteringModel = clusteringModel.predict(dataset.input);

stats = statsGen1turn.calculate_all(clusteringModel,dataset.input);

clusteringStatsNturns = clusteringStatsNturns.addResult(stats);

end

%% RESULTS / STATISTICS - CALCULATE

clusteringStatsNturns = clusteringStatsNturns.calculate_all();

%% RESULTS / STATISTICS - SHOW

% Clusters' Prototypes and Data (of last turn)
plotClustersAndData(dataset,clusteringModel);

%% SAVE VARIABLES AND VIDEO



%% END