function [DATAtr,DATAts] = hold_out_sysid(DATA,OPTIONS)

% --- Separates data between training and test ---
%
%   [DATAtr,DATAts] = hold_out_sysid(DATA,OPTIONS)
%
%   Input:
%       DATA.
%           input = input matrix             	[Nu x N]
%           output = output matrix            	[Ny x N]
%           lag_input = lag_u                   [1 x Nu]
%           lag_output = lag_y                  [1 x Ny]
%       OPTIONS.
%           ptrn = % of data for training     	[0 - 1]
%   Output:
%       DATAout.
%           DATAest = training samples (estimation)
%           DATApred = test samples (prediction)

%% INICIALIZAÇÕES

X = DATA.input;
Y = DATA.output;
[~,N] = size(Y);

ptrn = OPTIONS.ptrn;
Ntr = ceil(N*ptrn);

lag_input = DATA.lag_input;
lag_output = DATA.lag_output;

%% ALGORITMO

Xtr = X(:,1:Ntr);
ytr = Y(:,1:Ntr);
Xts = X(:,Ntr+1:end);
yts = Y(:,Ntr+1:end);

%% FILL STRUCTURE

DATAtr.input = Xtr;
DATAtr.output = ytr;
DATAtr.lag_input = lag_input;
DATAtr.lag_output = lag_output;

DATAts.input = Xts;
DATAts.output = yts;
DATAts.lag_input = lag_input;
DATAts.lag_output = lag_output;

%% END