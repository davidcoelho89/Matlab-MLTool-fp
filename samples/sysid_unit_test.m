%% MACHINE LEARNING TOOLBOX

% System Identification Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/02/03

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

OPT.Nr = 05;                % Number of realizations
OPT.alg = 'mlp';            % Which estimator will be used
OPT.prob = 'linear_arx';    % Linear ARX problem
OPT.lag_y = 2;              % Maximum lag of estimated outputs
OPT.lag_u = 1;              % Maximum lag of estimated inputs
OPT.add_noise = 1;          % "=1" if want to add noise
OPT.noise_var = 0.01;       % Noise variance
OPT.add_outlier = 0;        % "=1" if want to add outliers
OPT.outlier_ratio = 0.05;   % How many samples will be corrupted
OPT.outlier_ext = 0.5;      % Extension of signal that will be corrupted
OPT.norm = 0;               % Normalize input and outputs
OPT.prediction_type = 1;    % "=0": free simulation. ">0": n-steps ahead
OPT.ptrn = 0.5;             % Data used for training

OPT.input_type = 'prbs';    % Which type of input will be used
OPT.input_length = 500;     % Lengh of input
OPT.input_ts = [];          % Input time series for system models
OPT.y_coefs = [0.4,-0.6];   % Output coeficients for linear arx model
OPT.u_coefs = 2;            % Input coeficients for linear arx model

%% CHOOSE ALGORITHM HYPERPARAMETERS

% OLS
% HP.lambda = 0.001;

% LMS 
HP.Nep = 05;    % Numero de epocas
HP.eta = 0.1;	% Taxa de aprendizado

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

if(strcmp(OPT.prob,'linear_arx'))
    OPT.input_ts = build_input_ts(OPT); % Build input time series
end

DATAts = data_sysid_loading(OPT);       % Load input-output signals

DATA = regression_matrices(DATAts,OPT);	% Build regression matrices

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

OUTpre = regress_predict(DATApred,PAR);
STATS_pre_acc{r} = regress_stats_1turn(DATApred,OUTpre);

end

%% RESULTS / STATISTICS

nSTATS_estimation = regress_stats_nturns(STATS_est_acc);
nSTATS_prediction = regress_stats_nturns(STATS_pre_acc);

%% CONTROLLER

yh = OUTest.y_h;
y = DATAest.output;

figure;
plot(y,'b-')
hold on
plot(yh,'r-')
hold off

%% RESULTS / STATISTICS



%% END






























