function [PARout] = gp_train(DATA,PAR)

% --- Training of Gaussian Process Classifier ---
%
%   [PARout] = gp_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes          [p x N]
%           output = labels             [Nc x N]
%       PAR.
%           l2 = 
%           K = 
%           sig2 = 
%   Output:
%       PARout.
%           l2 = 
%           K = 
%           sig2 =           

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.l2 = 2;    	% GP constant 1
    PARaux.K = 1;      	% Kernel Type (gaussian = 1)
    PARaux.sig2 = 2;  	% GP constant 2
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'l2'))),
        PAR.l2 = 1;
    end
    if (~(isfield(PAR,'K'))),
        PAR.K = 1;
    end
    if (~(isfield(PAR,'sig2'))),
        PAR.sig2 = 1;
    end
end

%% INITIALIZATIONS

% ToDo - All

%% ALGORITHM

% ToDo - All

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.DATA = DATA;

%% END