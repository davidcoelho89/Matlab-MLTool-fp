%% MACHINE LEARNING TOOLBOX

% System Identification Algorithms (OOP Based) - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/12/22

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS AND OBJECTS

number_of_realizations = 10;    
percentage_for_training = 0.5;  
prediction_type = 1;            % "=0": free simulate. ">0": n-steps ahead
dataset_name = 'linear_arx_01'; % 'linear_arx_01' 'tank'
model_name = 'mlpLm';     	% 'mlp' 'lms' 'elm' 'elmOnline' 'rls' 'mlpLm'
normalization = 'zscore3';     	% 'none' 'zscore' 'zscore3'
output_lags = 2;                % [2,2];
input_lags = 2;                 % [2,2];

inputs_to_work_with = 'all';
outputs_to_work_with = 'all';

add_noise = 0;
noise_variance = 0.05;
noise_mean = 0;

add_outlier = 0;
outlier_rate = 0.01;
outlier_extension = 0.5;

normalizer = timeSeriesNormalizer();
normalizer.normalization = normalization;

statsGen1turn = regressionStatistics1turn();
sysId_stats_est = regressionStatisticsNturns(number_of_realizations);
sysId_stats_pre = regressionStatisticsNturns(number_of_realizations);

%% DATA LOADING AND PREPROCESSING

datasetTS = loadSysIdDataset(dataset_name);

% Visualize Time series (before noise, outliers, normalization)
plot_time_series(datasetTS);

% Select signals to work with
if(strcmp(dataset_name,'tank'))
    if(strcmp('all',outputs_to_work_with))
        % Does nothing
    elseif(outputs_to_work_with == 1)
        datasetTS.output = datasetTS.output(1,:);
    end
end

% Normalize time series
normalizer = normalizer.fit(datasetTS);
datasetTS = normalizer.transform(datasetTS);
% datasetTS = normalizer.reverse(datasetTS); % for debug

% Add noise to time series
if(add_noise)
    disp('Add Noise!');
    datasetTS.output = addTimeSeriesNoise(datasetTS.output,...
                                          noise_variance,...
                                          noise_mean);
end

% Add outliers to time series
if(add_outlier)
    disp('Add Outliers!');
    datasetTS.output = addTimeSeriesOutilers(datasetTS.output,...
                                             outlier_rate,...
                                             outlier_extension);
end

% Visualize Time series (after noise, outliers, normalization)
plot_time_series(datasetTS);

% Build Regression Matrices
dataset = buildRegressionMatrices(datasetTS,output_lags,input_lags);

% Divide data between train and test (estimate and predict)
[dataEst,dataPred] = splitSysIdDataset(dataset,percentage_for_training);

%% LOAD SYSTEM IDENTIFICATION MODEL AND CHOOSE ITS HYPERPARAMETERS

% Model's Object

model = initializeSysIdModel(model_name);

% General Hyperparameters

model.prediction_type = prediction_type;
model.output_lags = dataset.lag_output;

% Specific Hyperparameters

if(strcmp(model_name,'ols'))
    model.approximation = 'pinv';
    model.regularization = 0.0001;
    model.add_bias = 1;
    
elseif(strcmp(model_name,'lms'))
    model.number_of_epochs = 50;
    model.learning_step = 0.05;
    model.add_bias = 1;
    model.video_enabled = 0;
    
elseif(strcmp(model_name,'lmm'))
    model.number_of_epochs = 5;
    model.learning_step = 0.1;
    model.video_enabled = 0;
    model.add_bias = 1;
    model.Kout = 0.3;
    
elseif(strcmp(model_name,'rls'))
    model.forgiving_factor = 1;
    model.add_bias = 1;
    model.video_enabled = 0;
        
elseif(strcmp(model_name,'rlm'))
    model.number_of_epochs = 5;
        
elseif(strcmp(model_name,'mlp'))
    model.number_of_epochs = 30;            % 20 a 100
    model.number_of_hidden_neurons = 8;     % 2 a 20
    model.learning_rate = 0.05;             % 0.01 a 0.1
    model.moment_factor = 0.75;             % 0.5 a 0.09
    model.non_linearity = 'sigmoid';        % sigmoid ou tg_hyp ou relu
    model.add_bias = 1;
    model.video_enabled = 0;

elseif(strcmp(model_name,'mlpLm'))
    model.non_linearity = 'sigmoid';
    model.number_of_hidden_neurons = 7;
    model.minMSE = 1;
    model.minGRAD = 0.1;
    model.number_of_epochs = 1000;
    model.Muscale = 10;
    model.Mu_min = 1e-10;
    model.Mu_max = 1e+10;
    model.Mu_init = 0.01;
    model.add_bias = 0;
end

%% RUN EXPERIMENT

disp('Begin Algorithm');

for realization = 1:number_of_realizations

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(realization);
display(datestr(now));

% %%%%%%%%%%%% HYPERPARAMETER OPTIMIZATION %%%%%%%%%%%%%%%

% ToDo - All

% %%%%%%%%%%%%%%% SYSTEM'S ESTIMATION %%%%%%%%%%%%%%%%%%%%

model = model.fit(dataEst.input,dataEst.output);

% %%%%%%%% SYSTEM'S PREDICTION AND STATISTICS %%%%%%%%%%%%

% Predict Training data (in order to identify an overfit)
model = model.predict(dataEst.input);
stats = statsGen1turn.calculate_all(model.Yh,dataEst.output);
sysId_stats_est = sysId_stats_est.addResult(stats);

% Predict Test data (in order to identify Generalization ability)
model = model.predict(dataPred.input);
stats = statsGen1turn.calculate_all(model.Yh,dataPred.output);
sysId_stats_pre = sysId_stats_pre.addResult(stats);

end

%% RESULTS / STATISTICS - CALCULATE

sysId_stats_est = sysId_stats_est.calculate_all();
sysId_stats_pre = sysId_stats_pre.calculate_all();

%% RESULTS / STATISTICS - SHOW

y_est = sysId_stats_est.cell_of_results{1}.Y;
yh_est = sysId_stats_est.cell_of_results{1}.Yh;

figure;
plot(y_est(1,:),'b-')
title('Signal used for estimation')
hold on
plot(yh_est(1,:),'r--')
hold off

y_pred = sysId_stats_pre.cell_of_results{1}.Y;
yh_pred = sysId_stats_pre.cell_of_results{1}.Yh;

figure;
plot(y_pred(1,:),'b-')
title('Signal used for prediction')
hold on
plot(yh_pred(1,:),'r--')
hold off

%% END