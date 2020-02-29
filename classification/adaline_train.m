function [PARout] = adaline_train(DATA,PAR)

% --- Adaline classifier training ---
%
%   [PARout] = adaline_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix                   [p x N]
%           output = labels matrix                      [Nc x N]
%       PAR.
%           Ne = maximum number of epochs               [cte]
%           eta = learning rate                         [0.01 0.1]
%           Von = enable or disable video               [cte]
%   Output:
%       PARout.
%           W = Regression / Classification Matrix      [Nc x p+1]
%           MQEtr = mean quantization error of training	[1 x Ne]
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Ne = 200;       	% maximum number of training epochs
    PARaux.eta = 0.05;    	% Learning step
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
else
    if (~(isfield(PAR,'Ne'))),
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta'))),
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
end

%% ALGORITHM

[PARout] = lms_train(DATA,PAR); 

%% END