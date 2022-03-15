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
OPT.prediction_type = 1;    % "=0": free simulation. ">0": n-steps ahead

OPT.prob = 'tank';          % Which problem will be solved
OPT.prob2 = 01;             % Some especification of the problem

OPT.lag_y = 2;              % Maximum lag of estimated outputs
OPT.lag_u = 2;              % Maximum lag of estimated inputs

OPT.normalize = 0;          % "=1" if you want to normalize time series
OPT.norm_type = 1;          % Which type of normalization will be used

OPT.add_noise = 0;          % "=1" if you want to add noise
OPT.noise_var = 0.01;       % Noise variance

OPT.add_outlier = 0;        % "=1" if you want to add outliers
OPT.outlier_ratio = 0.05;   % How many samples will be corrupted
OPT.outlier_ext = 0.5;      % Extension of signal that will be corrupted

%% CHOOSE ALGORITHM HYPERPARAMETERS

OPT.alg = 'mlp';            % Which estimator will be used

% OLS
% HP.lambda = 0.001;

% LMS
% HP.Nep = 05;    % Numero de epocas
% HP.eta = 0.1;	% Taxa de aprendizado

% LMM
% HP.Nep = 05;    % Numero de epocas
% HP.eta = 0.1;	% Taxa de aprendizado
% HP.Kout = 0.3;	% Maximo nivel de erro

% MLP
HP.Nh = 5; % [5,3]; % Number of hidden neurons
HP.Ne = 100;       	% maximum number of training epochs
HP.eta = 0.05;    	% Learning step
HP.mom = 0.75;    	% Moment Factor
HP.Nlin = 2;       	% Non-linearity
HP.Von = 0;         % disable video 

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

% Test = Prediction -> Generate Prediction Errors
str_prediction = strcat(lower(OPT.alg),'_predict');
regress_predict = str2func(str_prediction);

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

% Load input-output signals
DATAts = data_sysid_loading(OPT);       

% Visualize Time series (before 
plot_time_series(DATAts);

% Select signals to work with
if(strcmp(OPT.prob,'tank'))
    DATAts.output = DATAts.output(1,:);
end

% Add noise to time series
if(OPT.add_noise)
    disp('Add Noise!');
end

% Add outliers to time series
if(OPT.add_outlier)
    disp('Add Outliers!');
end

% Normalize time series
if(OPT.normalize)
    disp('Normalize!');
end

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
plot(y_est,'b-')
title('Signal used for estimation')
hold on
plot(yh_est,'r-')
hold off

y_pred = DATApred.output;
yh_pred = OUTpred.y_h;

figure;
plot(y_pred,'b-')
title('Signal used for prediction')
hold on
plot(yh_pred,'r-')
hold off

%% CONTROLLER



%% RESULTS / STATISTICS



%% END






























