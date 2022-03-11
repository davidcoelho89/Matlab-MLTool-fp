%% IDENTIFICACAO DE SISTEMAS

% TC1 - Reconhecimento de Comandos de Voz
% Autor: David Nascimento Coelho
% Data: 24/02/2022

close;
clear;
clc;

%% GENERATE TIME SERIES (INPUT AND OUTPUT)

u_ts = [1,2,3,4,5,6,7,8,9,10];
y_ts = [11,12,13,14,15,16,17,18,19,20];

%% GENERATE REGRESSION MATRIX FROM TS (AR MODEL)

p = 2;  % Define AR order
[X,y] = regressionMatrixFromTS(y_ts,p);

disp(X);
disp(y);

%% GENERATE REGRESSION MATRIX FROM SISO (ARX MODEL)

lag_u = 2;  % Maximum input lag
lag_y = 2;  % Maximum output lag

[X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u);

disp(X);
disp(y);

%% END