function [OUT] = gp_classify(DATA,PAR)

% --- Test of Gaussian Process Classifier ---
%
%   [OUT] = gp_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                          [p x N]

%       PAR.
%           l2 = 
%           K = 
%           sig2 = 
%   Output:
%       OUT.
%           y_h = classifier's output                   [Nc x N]

%% INICIALIZAÇÕES

% ToDo - All (delete line above)

DATA.input = PAR;

%% ALGORITMO

% ToDo - All

%% FILL OUTPUT STRUCTURE

OUT.y_h = DATA.input;

%% END