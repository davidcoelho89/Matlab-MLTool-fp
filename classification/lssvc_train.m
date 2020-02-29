function [PARout] = lssvc_train(DATA,PAR)

% --- LSSVC classifier training ---
%
%   [PARout] = lssvc_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                                  [p x N]
%           output = labels                                     [Nc x N]
%       PAR.
%           lambda = regularization parameter                 	[cte]
%           epsilon = minimum value to be considered SV         [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PARout.
%           Xsv = attributes of support vectors                 [p x Nsv]
%           Ysv = labels of support vectors                    	[Nc x Nsv]
%           alpha = langrage multipliers                     	[Nc x Nsv]
%           b0 = optimum bias                                   [Nc x 1]
%           Nsv = number of support vectors                     [Nc x 1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.lambda = 0.5;  	% Regularization Constant
    PARaux.epsilon = 0;     % Minimum value of lagrange multipliers
    PARaux.Ktype = 2;       % Gaussian Kernel
    PARaux.sigma = 2;       % Gaussian Kernel std
    PAR = PARaux;
else
    if (~(isfield(PAR,'lambda'))),
        PAR.lambda = 0.5;
    end
    if (~(isfield(PAR,'epsilon'))),
        PAR.epsilon = 0;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 2;
    end
    if (~(isfield(PAR,'sigma'))),
        PAR.sigma = 2;
    end
end

%% INITIALIZATIONS

% Get Data
X = DATA.input;
Y = DATA.output;

% Get Parameters
lambda = PAR.lambda;
epsilon = PAR.epsilon;

% Initialize Problem
[~,N] = size(X);        % Number of samples
[Nc,~] = size(Y);       % Number of classes

% Initialize Outputs
Xsv = cell(1,Nc);
Ysv = cell(1,Nc);
alpha = cell(1,Nc);
b0 = cell(1,Nc);
Nsv = cell(1,Nc);

%% ALGORITHM

% Uses all training samples as support vectors
% For each class, one classifier (one vs all strategy)

for c = 1:Nc,

    % if it is a binary classifier, calculates the parameters only once
    if (c == 2 && Nc == 2),

        Xsv{2} = [];
        Ysv{2} = [];
        alpha{2} = [];
        b0{2} = [];
        Nsv{2} = [];

    else

        % Get labels of class c

        Yi = Y(c,:);

        % Calculate Kernel Matrix

        Omega = svc_f_kernel(X,Yi,PAR);

        % Build A matrix

        A = zeros(N+1,N+1);
        A(1,2:N+1) = Yi;
        A(2:N+1,1) = Yi';
        A(2:N+1,2:N+1) = Omega + (1/lambda)*eye(N);

        % Build b vector

        v = [0;ones(N,1)];

        % Solve the linear problem Ax = v:

        x_sys = linsolve(A,v);

        % Get bias, lagrange multipliers and support vectors

        b0{c} = x_sys(1);
        alpha{c} = x_sys(2:N+1);
        Xsv{c} = X;
        Ysv{c} = Yi;

        % Determines number of support vectors
        svi = find(abs(alpha{c}) > epsilon);
        Nsv{c} = length(svi);

    end

end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.Xsv = Xsv;
PARout.Ysv = Ysv;
PARout.alpha = alpha;
PARout.b0 = b0;
PARout.nsv = Nsv;

%% END