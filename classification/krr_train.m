 function [PARout] = krr_train(DATA,PAR)

% --- Kernel Ridge Classifier Training Function ---
%
%   PARout = krr_train(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       PAR.
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PARout.
%       	alphas = constants for the representer theorem      [N x Nc]
%           Xk = samples for the representer theorem            [p x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Ktype = 2;       % Kernel Type (Gaussian)
    PARaux.sig2n = 0.001;   % Kernel regularization parameter
    PARaux.sigma = 2;       % Kernel width
	PAR = PARaux;
else
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 2;
    end
    if (~(isfield(PAR,'sig2n'))),
        PAR.sig2n = 0.001;
    end
    if (~(isfield(PAR,'sigma'))),
        PAR.sigma = 2;
    end
end

%% INITIALIZATIONS

% Data Initialization

X = DATA.input;         % Input Matrix
Y = DATA.output;        % Output Matrix

% Get Hyperparameters

% (don't need. Just use HP inside the functions)

% Problem Initialization

[Nc,~] = size(Y);       % Total of Classes and Samples

% Init Outputs

alphas = cell(Nc,1);

%% ALGORITHM

Km = kernel_mat(X,PAR);

for c = 1:Nc,
    alphas{c} = pinv(Km)*Y(c,:)';
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.Xk = X;
PARout.alphas = alphas;

%% END