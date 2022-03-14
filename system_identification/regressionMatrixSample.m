%% IDENTIFICACAO DE SISTEMAS

% Sample - Building a Regression Matrix
% Autor: David Nascimento Coelho
% Data: 24/02/2022

close;
clear;
clc;

%% GENERATE TIME SERIES (INPUT AND OUTPUT)

u_ts1 = [1,2,3,4,5,6,7,8,9,10];
u_ts2 = [11,12,13,14,15,16,17,18,19,20];

y_ts1 = [21,22,23,24,25,26,27,28,29,30];
y_ts2 = [31,32,33,34,35,36,37,38,39,40];

%% GENERATE REGRESSION MATRIX FROM TS (AR MODEL)

DATAts.output = y_ts1;
OPT.lag_y = 2;

[DATA] = build_regression_matrices(DATAts,OPT);

disp(DATA.input);
disp(DATA.output);

%% GENERATE REGRESSION MATRIX FROM SISO (ARX MODEL)

clc; 

DATAts.input = u_ts1;
DATAts.output = y_ts1;

OPT.lag_u = 2;
OPT.lag_y = 2;

[DATA] = build_regression_matrices(DATAts,OPT);

disp(DATA.input);
disp(DATA.output);

%% GENERATE REGRESSION MATRIX FROM MISO (ARX MODEL)

clc; 

DATAts.input = [u_ts1; u_ts2];
DATAts.output = y_ts1;

OPT.lag_u = [2,2];
OPT.lag_y = 2;

[DATA] = build_regression_matrices(DATAts,OPT);

disp(DATA.input);
disp(DATA.output);

%% GENERATE REGRESSION MATRIX FROM SIMO (ARX MODEL)

clc; 

DATAts.input = u_ts1;
DATAts.output = [y_ts1; y_ts2];

OPT.lag_u = 2;
OPT.lag_y = [2,2];

[DATA] = build_regression_matrices(DATAts,OPT);

disp(DATA.input);
disp(DATA.output);

%% GENERATE REGRESSION MATRIX FROM MIMO (ARX MODEL)

clc; 

DATAts.input = [u_ts1; u_ts2];
DATAts.output = [y_ts1; y_ts2];

OPT.lag_u = [2,2];
OPT.lag_y = [2,2];

[DATA] = build_regression_matrices(DATAts,OPT);

disp(DATA.input);
disp(DATA.output);

%% END