function [OUT] = kqd_classify(DATA,PAR)

% --- Kernel Quadratic Discriminant Classify Function ---
%
%   [OUT] = kqd_classify(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
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
%       OUT.
%           y_h = classifier's output                           [Nc x N]

%% INITIALIZATION

% Data Initialization
X = DATA.input;                 % Input matrix
[~,N] = size(X);                % Number of samples

% Get Parameters

Ctype = PAR.Ctype;
sig2n = PAR.sig2n;

X_c = PAR.X_c;
n_c = PAR.n_c;
H_c = PAR.H_c;

Km = PAR.Km;
% Kinv = PAR.Kinv;
% Km_t = PAR.Km_t;
Kinv_t = PAR.Kinv_t;
% Km_reg = PAR.Km_reg;
% Kinv_reg = PAR.Km_inv_reg;
% Km_reg_t = PAR.Km_reg_t;
Kinv_reg_t = PAR.Kinv_reg_t;

% Problem Initilization
[Nc,~] = size(n_c);             % Number of prototypes and classes

% Init outputs
y_h = -1*ones(Nc,N);            % One output for each sample

%% ALGORITHM

for i = 1:N,
    % Get sample
    xi = X(:,i);
    % Init Discriminant
    gi = zeros(Nc,1);
    for c = 1:Nc,
        % Get Samples from class
        nc = n_c{c};
        Xc = X_c{c};
        % Get H matrix and Kernel Matrix from class
        H = H_c{c};
        Km_c = Km{c};
        % Calculate kx (and its centered version)
        kx = zeros(nc,1);
        for j = 1:nc,
            kx(j) = kernel_func(Xc(:,j),xi,PAR);
        end
        kx_t = H*(kx - (1/nc)*Km_c*ones(nc,1));
        % Calculate Mahalanobis Distance (Inverse Covariance)
        if (Ctype == 1)
            Kinv_t_c = Kinv_t{c};
            KMD = nc*kx_t'*Kinv_t_c*Kinv_t_c*kx_t;
        % Calculate Mahalanobis Distance (Regularized Covariance)
        elseif (Ctype == 2) 
            % Calculate kxx (and its centered version)
            kxx = kernel_func(xi,xi,PAR);
            kxx_t = kxx - (2/nc)*ones(1,nc)*kx ...
                        + (1/(nc^2))*ones(1,nc)*Km_c*ones(nc,1);
            Kinv_reg_t_c = Kinv_reg_t{c};
            KMD = (1/sig2n)*(kxx_t - kx_t'*Kinv_reg_t_c*kx_t);
        end
        % Calculate Eigenvalues of Km_c
        [~,L] = eig((1/nc)*Km_c);
        L = diag(L);
        % Discriminant Function
        gi(c) = -0.5 * (  KMD + log(prod(L)) );
    end
    y_h(:,i) = gi;
end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END