function [PARout] = svc_train(DATA,PAR)

% --- SVC classifier training ---
%
%   [PARout] = svc_train(DATA,PAR)
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
%           Ysv = labels of support vectors                     [Nc x Nsv]
%           alpha = langrage multipliers                        [Nc x Nsv]
%           b0 = optimum bias                                   [Nc x 1]
%           Nsv = number of support vectors                     [Nc x 1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.lambda = 5;      % Regularization Constant
    PARaux.epsilon = 0.001; % Minimum value of lagrange multipliers
    PARaux.Ktype = 2;       % Gaussian Kernel
    PARaux.sigma = 2;       % Gaussian Kernel std
    PAR = PARaux;
else
    if (~(isfield(PAR,'lambda'))),
        PAR.lambda = 5;
    end
    if (~(isfield(PAR,'epsilon'))),
        PAR.epsilon = 0.001;
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

% The number of support vectors varies according to each class
% For each class, one classifier (one vs all strategy)

% Uses the "Interior-Point-Convex" algorithm to solve QP problem
% Verify "Sequential-Minimal-Optimization" for big datasets

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

    Kmat = svc_f_kernel(X,Yi,PAR);

    % Quadratic optimization problem

    H = Kmat;               % Kernel Matrix with labels 
    f = -ones(N,1);         % Non-linear function
    Aeq = Yi;               % Equality Restriction
    beq = 0;                % Equality Restriction
    Aineq = zeros(1,N);     % Inequality Restriction
    bineq = 0;              % Inequality Restriction
    lb = zeros(N,1);        % Minimum values for SV (Lagrange Multipliers)
    ub = lambda*ones(N,1);	% Maximum values for SV (Lagrande Multipliers)
    x0 = [];                % Dosen't indicate a initial value for alphas

    opt = optimoptions(@quadprog,'Algorithm', ...
                       'interior-point-convex','Display','off');

    % Calculate alpha vector in order to identify support vectors

    [alphas,~,~,~,~] = quadprog(H,f,Aineq,bineq,Aeq,beq,lb,ub,x0,opt);

    sv_ind = find(abs(alphas) > epsilon); % indexes of support vectors

    % Get number of support vectors 

    Nsv{c} = length(sv_ind);

    % Get support vectors

    alpha{c} = alphas;
    Xsv{c} = X;
    Ysv{c} = Yi;

    % Calculate Optimum Bias

    b0{c} = svc_f_bias(Xsv{c},Ysv{c},alpha{c},PAR);

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