%% MACHINE LEARNING TOOLBOX

% System Identification Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/03/14

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

OPT.Nr = 10;                % Number of realizations
OPT.ptrn = 0.5;             % Data used for training
OPT.prediction_type = 0;    % "=0": free simulation. ">0": n-steps ahead

OPT.prob = 'tank';          % Which problem will be solved
OPT.prob2 = 01;             % Some especification of the problem

OPT.lag_y = 2;%[2,2];       % Maximum lag of estimated outputs
OPT.lag_u = 2;%[2,2];       % Maximum lag of estimated inputs

OPT.normalize = 0;          % "=1" if you want to normalize time series
OPT.norm_type = 4;          % Which type of normalization will be used

OPT.add_noise = 0;          % "=1" if you want to add noise
OPT.noise_var = 0.05;       % Noise variance
OPT.noise_mean = 0;         % Noise mean

OPT.add_outlier = 0;        % "=1" if you want to add outliers
OPT.outlier_rate = 0.01;    % Rate of samples that will be corrupted
OPT.outlier_ext = 0.5;      % Extension of signal that will be corrupted

OPT.alg = 'mlp';            % Which estimator will be used

%% CHOOSE ALGORITHM HYPERPARAMETERS

% General Hyperparameters

HP.prediction_type = OPT.prediction_type;

% Specific Hyperparameters

if(strcmp(OPT.alg,'ols'))
    HP.lambda = 0.001;
elseif(strcmp(OPT.alg,'lms'))
    HP.Nep = 10;        % Numero de epocas
    HP.eta = 0.1;       % Taxa de aprendizado
    HP.add_bias = 1;    % Adiciona ou nao o bias
elseif(strcmp(OPT.alg,'lmm'))
    HP.Nep = 05;    % Numero de epocas
    HP.eta = 0.1;	% Taxa de aprendizado
    HP.Kout = 0.3;	% Maximo nivel de erro
elseif(strcmp(OPT.alg,'rls'))
    % ToDo
elseif(strcmp(OPT.alg,'rlm'))
    % ToDo
elseif(strcmp(OPT.alg,'mlp'))
    HP.Nh = [5,3]; % 8; % Number of hidden neurons
    HP.Ne = 200;       	% maximum number of training epochs
    HP.eta = 0.05;    	% Learning step
    HP.mom = 0.75;    	% Moment Factor
    HP.Nlin = 2;       	% Non-linearity
    HP.Von = 0;         % disable video
end

%% ACCUMULATORS

NAMES = {'estimation','prediction'};
STATS_est_acc = cell(OPT.Nr,1);   	% Acc of Statistics of estimation data
STATS_pre_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

%% HANDLERS FOR REGRESSION FUNCTIONS

algorithm_name = upper(OPT.alg);

% Training = Estimation -> Generate Residues
str_estimation = strcat(lower(OPT.alg),'_estimate');
regress_estimate = str2func(str_estimation);
clear str_estimation;

% Test = Prediction -> Generate Prediction Errors
str_prediction = strcat(lower(OPT.alg),'_predict');
regress_predict = str2func(str_prediction);
clear str_prediction;

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

% Load input-output signals
DATAts = data_sysid_loading(OPT);

% Visualize Time series (before noise, outliers, normalization)
plot_time_series(DATAts);

% Select signals to work with
if(strcmp(OPT.prob,'tank'))
    if(OPT.prob2 == 1)
        DATAts.output = DATAts.output(1,:);
    end
    if(OPT.prob2 == 2)
        DATAts.input = [DATAts.input;DATAts.input];
    end
end

% Normalize time series
if(OPT.normalize)
    disp('Normalize!');
    PARnorm = normalizeTimeSeries_fit(DATAts,OPT);
    DATAts = normalizeTimeSeries_transform(DATAts,PARnorm);
end

% Add noise to time series
if(OPT.add_noise)
    disp('Add Noise!');
    DATAts.output = addTimeSeriesNoise(DATAts.output,OPT.noise_var, ...
                                          OPT.noise_mean);
end

% Add outliers to time series
if(OPT.add_outlier)
    disp('Add Outliers!');
    DATAts.output = addTimeSeriesOutilers(DATAts.output,OPT);
end

% Visualize Time series (after noise, outliers, normalization)
plot_time_series(DATAts);

% Build Regression Matrices
DATA = build_regression_matrices(DATAts,OPT);	

% Divide data between train and test (estimate and predict)
[DATAest,DATApred] = hold_out_sysid(DATA,OPT);

%% SYS ID - HOLD OUT / NORMALIZE / ESTIMATE / PREDICT

disp('Begin Algorithm');

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(r);
display(datestr(now));

% %%%%%%%%%%%%%%% SYSTEM'S ESTIMATION %%%%%%%%%%%%%%%%%%%%

PAR = regress_estimate(DATAest,HP);

% %%%%%%%% SYSTEM'S PREDICTION AND STATISTICS %%%%%%%%%%%%

OUTest = regress_predict(DATAest,PAR);
STATS_est_acc{r} = regress_stats_1turn(DATAest,OUTest);

OUTpred = regress_predict(DATApred,PAR);
STATS_pre_acc{r} = regress_stats_1turn(DATApred,OUTpred);

end

%% RESULTS / STATISTICS

nSTATS_estimation = regress_stats_nturns(STATS_est_acc);
nSTATS_prediction = regress_stats_nturns(STATS_pre_acc);

y_est = DATAest.output;
yh_est = OUTest.y_h;

figure;
plot(y_est(1,:),'b-')
title('Signal used for estimation')
hold on
plot(yh_est(1,:),'r--')
legend('real','estimated')
hold off

y_pred = DATApred.output;
yh_pred = OUTpred.y_h;

figure;
plot(y_pred(1,:),'b-')
title('Signal used for prediction')
hold on
plot(yh_pred(1,:),'r--')
legend('real','estimated')
hold off

%% CONTROLLER



%% RESULTS / STATISTICS



%% END






























