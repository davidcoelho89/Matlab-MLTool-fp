%% MACHINE LEARNING TOOLBOX

% System Identification Algorithms (OOP Based) - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/03/14

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS AND OBJECTS

number_of_realizations = 10;    
percentage_for_training = 0.5;  
prediction_type = 1;            % "=0": free simulate. ">0": n-steps ahead
dataset_name = 'tank';         
model_name = 'lms';             
normalization = 'zscore3';      
output_lags = [2,2];             
input_lags = [2,2];              

inputs_to_work_with = 'all';
outputs_to_work_with = 1;

add_noise = 0;
noise_variance = 0.05;
noise_mean = 0;

add_outlier = 0;
outlier_rate = 0.01;
outlier_extension = 0.5;

normalizer = timeSeriesNormalizer();
normalizer.normalization = normalization;

statsGen1turn = regressionStatistics1turn();
sysId_stats_est = regressionStatisticsNturns();
sysId_stats_pre = regressionStatisticsNturns();

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
[DATAest,DATApred] = splitSysIdDataset(dataset,percentage_for_training);

%% LOAD SYSTEM IDENTIFICATION MODEL AND CHOOSE ITS HYPERPARAMETERS

% Model's Object
lmsArx
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
    model.number_of_epochs = 5;
    model.learning_step = 0.05;
    model.video_enabled = 0;
    model.add_bias = 1;
    
elseif(strcmp(model_name,'lmm'))
    model.number_of_epochs = 5;
    model.learning_step = 0.1;
    model.video_enabled = 0;
    model.add_bias = 1;
    model.Kout = 0.3;
    
elseif(strcmp(model_name,'rls'))
    model.number_of_epochs = 5;
        
elseif(strcmp(model_name,'rlm'))
    model.number_of_epochs = 5;
        
elseif(strcmp(model_name,'mlp'))
    model.number_of_epochs = 200;
    model.number_of_hidden_neurons = 8;
    model.learning_rate = 0.05;
    model.moment_factor = 0.75;
    model.non_linearity = 'sigmoid';
    model.add_bias = 1;
    model.video_enabled = 0;

end

%% SYS ID - HOLD OUT / NORMALIZE / ESTIMATE / PREDICT

disp('Begin Algorithm');

for r = 1:number_of_realizations

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(r);
display(datestr(now));

% %%%%%%%%%%%%%%% SYSTEM'S ESTIMATION %%%%%%%%%%%%%%%%%%%%

model = model.fit(DATAest.input,DATAest.output);

% %%%%%%%% SYSTEM'S PREDICTION AND STATISTICS %%%%%%%%%%%%

model = model.predict(DATAest.input);
stats = statsGen1turn(model.Yh,DATAest.output);
sysId_stats_est = sysId_stats_est.add(stats);

model = model.predict(DATApred.input);
stats = statsGen1turn(model.Yh,DATAest.output);
sysId_stats_pre = sysId_stats_pre.add(stats);

end

sysId_stats_est = sysId_stats_est.calculate_all();

sysId_stats_pre = sysId_stats_pre.calculate_all();

%% RESULTS / STATISTICS
% 
% nSTATS_estimation = regress_stats_nturns(STATS_est_acc);
% nSTATS_prediction = regress_stats_nturns(STATS_pre_acc);
% 
% y_est = DATAest.output;
% yh_est = OUTest.y_h;
% 
% figure;
% plot(y_est(1,:),'b-')
% title('Signal used for estimation')
% hold on
% plot(yh_est(1,:),'r-')
% hold off
% 
% y_pred = DATApred.output;
% yh_pred = OUTpred.y_h;
% 
% figure;
% plot(y_pred(1,:),'b-')
% title('Signal used for prediction')
% hold on
% plot(yh_pred(1,:),'r-')
% hold off
% 
%% CONTROLLER



%% RESULTS / STATISTICS



%% END