%% MACHINE LEARNING TOOLBOX

% System Identification Algorithms (OOP Based) - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/03/14

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

number_of_realizations = 10;    
percentage_for_training = 0.5;  
prediction_type = 1;            % "=0": free simulate. ">0": n-steps ahead
dataset_name = 'motor';         
model_name = 'lms';             
output_lags = [2,2];             
input_lag = [2,2];              
normalization = 'zscore3';      

add_noise = 0;
noise_variance = 0.05;
noise_mean = 0;

add_outlier = 0;
outlier_rate = 0.01;
outiler_extension = 0.5;

%% LOAD SYSTEM IDENTIFICATION MODEL AND CHOOSE ITS HYPERPARAMETERS

model = initializeSysIdModel(model_name);

model.prediction_type = prediction_type;
model.output_lags = output_lags;

if(strcmp(model_name,'ols'))
    classifier.approximation = 'pinv';
    classifier.regularization = 0.0001;
    classifier.add_bias = 1;

elseif(strcmp(model_name,'lms'))
    model.number_of_epochs = 200;
    model.learning_step = 0.05;
    model.video_enabled = 0;
    model.add_bias = 1;

elseif(strcmp(model_name,'lmm'))
    model.number_of_epochs = 200;
    model.learning_step = 0.1;
    model.video_enabled = 0;
    model.add_bias = 1;
    model.Kout = 0.3;

elseif(strcmp(model_name,'rls'))
    

elseif(strcmp(model_name,'rlm'))
    

elseif(strcmp(model_name,'mlp'))
        model.number_of_epochs = 200;
        model.number_of_hidden_neurons = 8;
        model.learning_rate = 0.05;
        model.moment_factor = 0.75;
        model.non_linearity = 'sigmoid';
        model.add_bias = 1;
        model.video_enabled = 0;
        model.prediction_type = prediction_type;
        model.output_lags = [];
end

%% CHOOSE ALGORITHM HYPERPARAMETERS

% MLP

% 
% %% ACCUMULATORS
% 
% NAMES = {'estimation','prediction'};
% STATS_est_acc = cell(number_of_realizations,1);	% Acc of Stats of estimation data
% STATS_pre_acc = cell(number_of_realizations,1);	% Acc of Stats of test data
% nSTATS_all = cell(2,1);                         % Acc of General statistics
% 
% %% HANDLERS FOR REGRESSION FUNCTIONS
% 
% algorithm_name = upper(OPT.alg);
% 
% % Training = Estimation -> Generate Residues
% str_estimation = strcat(lower(OPT.alg),'_estimate');
% regress_estimate = str2func(str_estimation);
% 
% % Test = Prediction -> Generate Prediction Errors
% str_prediction = strcat(lower(OPT.alg),'_predict');
% regress_predict = str2func(str_prediction);
% 
% %% DATA LOADING, PRE-PROCESSING, VISUALIZATION
% 
% % Load input-output signals
% DATAts = data_sysid_loading(OPT);
% 
% % Visualize Time series (before noise, outliers, normalization)
% plot_time_series(DATAts);
% 
% % Select signals to work with
% if(strcmp(OPT.prob,'tank'))
%     if(OPT.prob2 == 1)
%         DATAts.output = DATAts.output(1,:);
%     end
%     if(OPT.prob2 == 2)
%         DATAts.input = [DATAts.input;DATAts.input];
%     end
% end
% 
% % Normalize time series
% if(OPT.normalize)
%     disp('Normalize!');
%     PARnorm = normalizeTimeSeries_fit(DATAts,OPT);
%     DATAts = normalizeTimeSeries_transform(DATAts,PARnorm);
% end
% 
% % Add noise to time series
% if(OPT.add_noise)
%     disp('Add Noise!');
%     DATAts.output = addTimeSeriesNoise(DATAts.output,OPT);
% end
% 
% % Add outliers to time series
% if(OPT.add_outlier)
%     disp('Add Outliers!');
%     DATAts.output = addTimeSeriesOutilers(DATAts.output,OPT);
% end
% 
% % Visualize Time series (after noise, outliers, normalization)
% plot_time_series(DATAts);
% 
% % Build Regression Matrices
% DATA = build_regression_matrices(DATAts,OPT);	
% 
% % Divide data between train and test (estimate and predict)
% [DATAest,DATApred] = hold_out_sysid(DATA,OPT);
% 
% %% SYS ID - HOLD OUT / NORMALIZE / ESTIMATE / PREDICT
% 
% disp('Begin Algorithm');
% 
% for r = 1:number_of_realizations
% 
% % %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%
% 
% disp('Turn and Time');
% disp(r);
% display(datestr(now));
% 
% % %%%%%%%%%%%%%%% SYSTEM'S ESTIMATION %%%%%%%%%%%%%%%%%%%%
% 
% PAR = regress_estimate(DATAest,HP);
% 
% % %%%%%%%% SYSTEM'S PREDICTION AND STATISTICS %%%%%%%%%%%%
% 
% OUTest = regress_predict(DATAest,PAR);
% STATS_est_acc{r} = regress_stats_1turn(DATAest,OUTest);
% 
% OUTpred = regress_predict(DATApred,PAR);
% STATS_pre_acc{r} = regress_stats_1turn(DATApred,OUTpred);
% 
% end
% 
% %% RESULTS / STATISTICS
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
% %% CONTROLLER
% 
% 
% 
% %% RESULTS / STATISTICS
% 
% 
% 
% %% END