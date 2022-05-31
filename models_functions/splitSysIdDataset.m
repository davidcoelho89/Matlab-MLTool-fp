function [DATAtr,DATAts] = splitSysIdDataset(DATA,percentage_for_training)

% --- Separates data between training and test ---
%
%   [DATAtr,DATAts] = splitSysIdDataset(DATA,percentage_for_training)
%
%   Input:
%       DATA.
%           input = input matrix             	[Nu x N]
%           output = output matrix            	[Ny x N]
%           lag_input = lag_u                   [1 x Nu]
%           lag_output = lag_y                  [1 x Ny]
%       percentage_for_training                 [0 - 1]
%   Output:
%       DATAout.
%           DATAtr = training samples (estimation)
%           DATAts = test samples (prediction)

%% INICIALIZAÇÕES

[~,N] = size(DATA.output);
Ntr = ceil(N*percentage_for_training);

%% ALGORITMO

DATAtr.input = DATA.input(:,1:Ntr);
DATAtr.output = DATA.output(:,1:Ntr);
DATAtr.lag_input = DATA.lag_input;
DATAtr.lag_output = DATA.lag_output;

DATAts.input = DATA.input(:,Ntr+1:end);
DATAts.output = DATA.output(:,Ntr+1:end);
DATAts.lag_input = DATA.lag_input;
DATAts.lag_output = DATA.lag_output;

%% END