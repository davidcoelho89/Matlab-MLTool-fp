 function [PAR] = kqd_train(DATA,HP)

% --- Kernel Quadratic Discriminant Training Function ---
%
%   PAR = kqd_train(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       PAR.
%           Ctype = type of quadratic classifier              	[cte]
%               1: Inverse Covariance
%               2: Regularized Covariance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PARout.
%       	H = centering matrix                       [Nc x nc x nc]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    PARaux.Ctype = 2;       % Classifier type (regularized)
    PARaux.Ktype = 2;       % Kernel Type (Gaussian)
    PARaux.sig2n = 0.001;   % Kernel regularization parameter
    PARaux.sigma = 2;       % Kernel width
	HP = PARaux;
else
    if (~(isfield(HP,'Ctype'))),
        HP.Ctype = 2;
    end
    if (~(isfield(HP,'Ktype'))),
        HP.Ktype = 2;
    end
    if (~(isfield(HP,'sig2n'))),
        HP.sig2n = 0.001;
    end
    if (~(isfield(HP,'sigma'))),
        HP.sigma = 2;
    end
end

%% INITIALIZATIONS

% Data Initialization

X = DATA.input;         % Input Matrix
Y = DATA.output;        % Output Matrix

% Get Hyperparameters

sig2n = HP.sig2n;

% Problem Initialization

[Nc,~] = size(Y);       % Number of samples and classes
[~,Y_seq] = max(Y);     % Sequential labels

% Init Outputs

X_c = cell(Nc,1);           % Hold Data points of each class
n_c = cell(Nc,1);           % Hold Number of samples of each class
H_c = cell(Nc,1);           % Hold H matrix of each class

Km = cell(Nc,1);            % kernel matrix
Kinv = cell(Nc,1);          % inverse kernel matrix
Km_t = cell(Nc,1);          % "tilde" -> "centered"
Kinv_t = cell(Nc,1);        % "tilde" -> "centered"
Km_reg = cell(Nc,1);        % regularized kernel matrix
Kinv_reg = cell(Nc,1);      % regularized inverse kernel matrix
Km_reg_t = cell(Nc,1);      % "tilde" -> "centered"
Kinv_reg_t = cell(Nc,1);    % "tilde" -> "centered"

%% ALGORITHM

for c = 1:Nc,
    % Get samples of class
    n_c{c} = sum(Y_seq == c);
    X_c{c} = X(:,(Y_seq == c));
    % Calculate H Matrix of class
    V1 = ones(n_c{c},1);
    Ident = eye(n_c{c});
    H_c{c} = (Ident - (1/n_c{c})*(V1*V1'));
    % Calculate Kernel Matrix (and its centered version)
	Km{c} = kernel_mat(X_c{c},HP);
    Kinv{c} = pinv(Km{c});
    Km_t{c} = H_c{c}*Km{c}*H_c{c};
    Kinv_t{c} = pinv(Km_t{c});
    % Calculate Regularized Kernel Matrix (and its centered version)
    Km_reg{c} = kernel_mat(X_c{c},HP) + (n_c{c} - 1)*sig2n*eye(n_c{c});
    Kinv_reg{c} = pinv(Km_reg{c});
	Km_reg_t{c} = H_c{c}*Km_reg{c}*H_c{c};
    Kinv_reg_t{c} = pinv(Km_reg_t{c});
end

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.X_c = X_c;
PAR.n_c = n_c;
PAR.H_c = H_c;
PAR.Km = Km;
PAR.Kinv = Kinv;
PAR.Km_t = Km_t;
PAR.Kinv_t = Kinv_t;
PAR.Km_reg = Km_reg;
PAR.Kinv_reg = Kinv_reg;
PAR.Km_reg_t = Km_reg_t;
PAR.Kinv_reg_t = Kinv_reg_t;

%% END