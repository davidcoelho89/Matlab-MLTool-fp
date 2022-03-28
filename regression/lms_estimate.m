function [PARout] = lms_estimate(DATA,PAR)

% --- LMS System Identification Estimation ---
%
%   [PARout] = lms_estimate(DATA,PAR)
%
%   Input:
%       DATA.
%           input = inputs matrix                           [p x N]
%           output = outputs matrix                         [No x N]
%       PAR.
%           Ne = maximum number of epochs                   [cte]
%           eta = learning rate                             [0.01 0.1]
%           add_bias = whether or not to add the bias       [0 or 1]
%           Von = enable or disable video                   [cte]
%   Output:
%       PARout.
%           W = Regression Matrix                           [Nc x p+1]
%           W_acc = See parameters evolution                [Nep*N+1]
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.Ne = 200;            % maximum number of training epochs
    PARaux.eta = 0.05;          % Learning step
    PARaux.add_bias = 1;        % Add bias
    PARaux.Von = 0;             % disable video 
    PARaux.prediction_type = 1; % 1-step ahead prediction
    PAR = PARaux;
else
    if (~(isfield(PAR,'Ne')))
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta')))
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'add_bias')))
        PAR.add_bias = 1;
    end
    if (~(isfield(PAR,'Von')))
        PAR.Von = 0;
    end
    if (~(isfield(PAR,'prediction_type')))
        PAR.prediction_type = 1;
    end
end

%% INITIALIZATIONS

Xtr = DATA.input;
ytr = DATA.output;

[p,N] = size(Xtr);
[No,~] = size(ytr);

Nep = PAR.Nep;
eta = PAR.eta;
add_bias = PAR.add_bias;
% Von = PAR.Von;              % enable or disable video

if(add_bias)
    W = 0.01*rand(No,p+1);
    Xtr = [ones(1,N); Xtr];
else
    W = 0.01*rand(No,p);
end

W_acc = cell(1,Nep*N+1);
W_acc{1,1} = W;
cont = 1;

VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep

    for n = 1:N

        xn = Xtr(:,n);      % Get sample vector
        yh = W * xn;        % Get outputs
        er = ytr(:,n) - yh; % Error for each output
        
        W = W + eta * er * xn' / (xn'*xn);

        cont = cont+1;
        W_acc{1,cont} = W;
    end

end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.W_acc = W_acc;
PARout.VID = VID;
PARout.lag_output = DATA.lag_output;

%% END