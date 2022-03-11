function [DATAtr,DATAts] = hold_out_sysid(DATA,OPTIONS)

% --- Separates data between training and test ---
%
%   [DATAtr,DATAts] = hold_out_sysid(DATA,OPTIONS)
%
%   Input:
%       DATA.
%           input = input matrix                    [Nu x N]
%           output = output matrix                  [Ny x N]
%       OPTIONS.
%           ptrn = % of data for training           [0 - 1]
%   Output:
%       DATAout.
%           DATAest = training samples (estimation)
%           DATApred = test samples (prediction)

%% INICIALIZAÇÕES

X = DATA.input;
y = DATA.output;
N = length(y);

ptrn = OPTIONS.ptrn;
Ntr = ceil(N*ptrn);

%% ALGORITMO

Xtr = X(1:Ntr,:);
ytr = y(1:Ntr,:);
Xts = X(Ntr+1:end,:);
yts = y(Ntr+1:end,:);

%% FILL STRUCTURE

DATAtr.input = Xtr;
DATAtr.output = ytr;
DATAts.input = Xts;
DATAts.output = yts;

%% END