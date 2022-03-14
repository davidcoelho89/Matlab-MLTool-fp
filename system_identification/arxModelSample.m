%% IDENTIFICACAO DE SISTEMAS

% Sample - Building input and outputs for ARX Models
% Autor: David Nascimento Coelho
% Data: 24/02/2022

close;
clear;
clc;

%% BUILD INPUT SIGNAL

% Hyperparameters
HP.input_type = 'prbs';         % What signal will be generated
                            	% ('awgn' ; 'prbs_prob' ; 'aprbs')
HP.input_length = 500;          % Signal length
HP.u_mean = 0;                  % Signal mean (if needed)
HP.u_var = 0.5;                 % Signal variance (if needed)
HP.prob = 0.7;                  % Probability of u(n) = 1 (if needed)

% Build Input Signal (time series)
u_ts = build_input_ts(HP);

% Build Samples Vector
n = 1:length(u_ts);

% Plot Signal
figure;
plot(n,u_ts,'r-')
title('Input Time Series')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(u_ts)-0.1,max(u_ts)+0.1])

%% BUILD OUTPUT SIGNAL

% Hyperparameters
HP.y_coefs = [0.4,-0.6];    % Output coefficients for linear arx model
HP.u_coefs = 2;             % Input coefficients for linear arx model
HP.noise_var = 0;           % Variance of noise added to output

% Build Output Signal (time series)
% y[n] = 0.4*y[n-1] -0.6*y[n-2] + 2*u[n-1]
y_ts = arxOutputFromInput(u_ts,HP);

% Plot Output Signal
% Plot Signal
figure;
plot(n,y_ts,'r-')
title('Output Time Series')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(y_ts)-0.1,max(y_ts)+0.1])

%% SAVE SYSTEM INPUT-OUTPUT

% savefile = 'linear_arx_01.mat';
% save(savefile,'u_ts','y_ts');

%% END