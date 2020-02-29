function [PAR] = kpca_feature(DATA,HP)

% --- Kernel Principal Component Analysis for Feature Selection ---
%
%   PAR = kpca_feature(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       HP.
%           tol = tolerance of explained value  [0 - 1]         [cte]
%           rem = mean removal [0 or 1]                         [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PAR.
%           mu = mean of input matrix                           [p x 1]
%           q = number of used attributes                       [cte]
%           input = attributes matrix                           [q x N]
%           output = labels matrix                              [Nc x N]
%           L = eigenvalues of cov                              [1 x p]
%           V = eigenvectors of cov                             [p x p]
%           W = Transformation Vectors                          [p x q]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    PARaux.tol = 0.95;
    PARaux.rem = 1;
    PARaux.alg = 1;
    PARaux.Ktype = 2;
    PARaux.sig2n = 0.001;
    PARaux.sigma = 2;
    HP = PARaux;
else
    if (~(isfield(HP,'tol'))),
        HP.tol = 0.95;
    end
    if (~(isfield(HP,'rem'))),
        HP.rem = 1;
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

X = DATA.input;     % input matrix
Y = DATA.output;    % output matrix
[p,N] = size(X);    % dimensions of input matrix

tol = HP.tol;   	% explained value
rem = HP.rem;       % remove or not mean

%% ALGORITHM

% Calculate Mean and Remove it from each sample 

Xmean = mean(X,2);
if (rem == 1),
    X = X - repmat(Xmean,1,N);
end

% Calculate Kernel Matrix

Km = kernel_mat(X,HP);

% Get eigenvectors and eigenvalues of Kernel matrix

[V,L] = eig(Km);
L = diag(L);

% Sort eigenvalues and eigenvectors

SORT = bubble_sort(L,2);    % uses eigenvalues to define the order
L = L(SORT.ind);            % sort eigenvalues
V = V(:,SORT.ind);          % sort eigenvectors

% Explained variance

Ev = zeros(1,N);
Ev(1) = L(1);
for i = 2:N,
    Ev(i) = Ev(i-1) + L(i);
end
Ev = Ev/sum(L);

% Find number of Principal Components

for i = 1:N,
    if(Ev(i) >= tol || i == p),
        q = i;
        break;
    end
end

% Get transformation matrix [p x q]

A = V(:,1:q);

% Transform input matrix

Xtr = zeros(q,N);

for n = 1:N,
    xn = X(:,n);
    for k = 1:q,
        ak = A(:,k);
        for i = 1:N,
            xi = X(:,i);
            Xtr(k,n) = Xtr(k,n) + ak(i)*kernel_func(xn,xi,HP);
        end
    end
end

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.mu = Xmean;
PAR.input = Xtr;
PAR.X = X;
PAR.output = Y;
PAR.L = L;
PAR.V = V;
PAR.W = A;
PAR.q = q;
PAR.Ev = Ev;

%% END