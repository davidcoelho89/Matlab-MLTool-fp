function [PARout] = lms_estimate(DATA,PAR)

% --- LMS System Identification Estimation ---
%
%   [PARout] = lms_estimate(DATA,PAR)
%
%   Input:
%       DATA.
%           input = inputs matrix                           [p x N]
%           output = outputs matrix                         [Nc x N]
%       PAR.
%           Ne = maximum number of epochs                   [cte]
%           eta = learning rate                             [0.01 0.1]
%           Von = enable or disable video                   [cte]
%   Output:
%       PARout.
%           W = Regression Matrix                           [Nc x p+1]
%           W_acc = See parameters evolution
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.Ne = 200;       	% maximum number of training epochs
    PARaux.eta = 0.05;    	% Learning step
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
else
    if (~(isfield(PAR,'Ne')))
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta')))
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'Von')))
        PAR.Von = 0;
    end
end

%% INITIALIZATIONS

Xtr = DATA.input;
ytr = DATA.output;

[N,p] = size(Xtr);

Nep = PAR.Nep;
eta = PAR.eta;
% Von = PAR.Von;              % enable or disable video

W = randn(p,1);

W_acc = zeros(p,Nep*N+1);
W_acc(:,1) = W;
cont = 1;

VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep

    for n = 1:N
        xn = Xtr(n,:)';
        yh = W' * xn;
        er = ytr(n) - yh;
        W = W + eta * er * xn / (xn'*xn);

        cont = cont+1;
        W_acc(:,cont) = W;
    end

end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.W_acc = W_acc;
PARout.VID = VID;

%% END