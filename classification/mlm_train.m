function [PARout] = mlm_train(DATA,PAR)

% --- MLM Classifier Training ---
%
%   [PARout] = mlm_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = training attributes             [p x N]
%           output = training labels                [Nc x N]
%       PAR.
%           dist = type of distance                 [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance  
%               2: Euclidean distance
%           Ktype = Kernel Type                    	[cte]
%               = 0 -> non-kernelized algorithm
%           K = number of reference points          [cte]
%   Output:
%       PARout.
%           B = Regression matrix                   [K x K]
%           Rx = Input reference points             [K x p]
%           Ty = Output reference points            [K x Nc]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.dist = 2; 	% Gaussian distance
    PARaux.Ktype = 0; 	% Non-kernelized Algorithm
    PARaux.K = 9;       % Number of reference points
    PAR = PARaux;
else
    if (~(isfield(PAR,'dist')))
        PAR.dist = 2;
    end
    if (~(isfield(PAR,'Ktype')))
        PAR.Ktype = 0;
    end
    if (~(isfield(PAR,'K')))
        PAR.K = 9;
    end
end

%% INITIALIZATIONS

% Get Data
X = DATA.input';    % Input matrix  [N x p]
Y = DATA.output';   % Output matrix [N x Nc]
[N,~] = size(X);    % Number of samples

% Get hyperparameters
K = PAR.K;          % Number of reference points

%% ALGORITHM

% Aleatory selects K reference points

I = randperm(N);
Rx = X(I(1:K),:);
Ty = Y(I(1:K),:);

% Input distance matrix of reference points

Dx = zeros(N,K);                         	% Init Dx [N x K]

for n = 1:N
    xi = X(n,:);                         	% Get input sample
    for k = 1:K
        mk = Rx(k,:);                      	% Get input reference point
        Dx(n,k) = vectors_dist(xi,mk,PAR);	% Calculates distance
    end
end

% Output distance matrix of reference points

Dy = zeros(N,K);                         	% Init Dy [N x K]

for n = 1:N
    yi = Y(n,:);                            % Get output sample
    for k = 1:K
        tk = Ty(k,:);                       % Get output reference point   
        Dy(n,k) = vectors_dist(yi,tk,PAR);	% Calculates distance
    end
end

% Calculates Regeression Matrix [K x K]

B = pinv(Dx)*Dy;

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.B = B;
PARout.Rx = Rx;
PARout.Ty = Ty;

%% END