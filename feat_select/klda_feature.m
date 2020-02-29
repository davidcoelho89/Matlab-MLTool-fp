function [PAR] = klda_feature(DATA,HP)

% --- Kernel Linear Discriminant Analysis for Feature Selection ---
%
%   PAR = klda_feature(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       PAR.
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
%       PARout.
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

X = DATA.input;             % input matrix
Y = DATA.output;            % output matrix
[p,L] = size(X);            % dimensions of input matrix
[~,Y_seq] = max(Y);         % Classes (sequential)
Nc = length(unique(Y_seq)); % Nubmer of classes of the problem

tol = HP.tol;               % explained value
rem = HP.rem;               % remove or not mean
sig2n = HP.sig2n;           % regularization parameter

%% ALGORITHM

% Calculate Mean and Remove it from each sample 

Xmean = mean(X,2);
if (rem == 1),
    X = X - repmat(Xmean,1,L);
end

% Calculate Kernel Matrices

M = zeros(L,L);
N = zeros(L,L);

Xi = cell(Nc,1);
li = cell(Nc,1);
Mi = cell(Nc,1);
Ni = cell(Nc,1);

% Calculate M asterisk
Mst = zeros(L,1);
for j = 1:L,
    xj = X(:,j);
    for k = 1:L,
        xk = X(:,k);
        Mst(j) = Mst(j) + kernel_func(xj,xk,HP);
    end
end
Mst = Mst/L;

for c = 1:Nc,
    % get samples of classes
    Xi{c} = X(:,(Y_seq == c));
    li{c} = length(find(Y_seq == c));
    % get Mi matrix
    Mi{c} = zeros(L,1);
    for j = 1:L,
        xj = X(:,j);
        for k = 1:li{c},
            xk = Xi{c}(:,k);
            Mi{c}(j) = Mi{c}(j) + kernel_func(xj,xk,HP);
        end
    end
    Mi{c} = Mi{c}/li{c};
    % get Ni matrix
    Kc = zeros(L,li{c});
    for n = 1:L,
        xn = X(:,n);
        for m = 1:li{c},
            xm = Xi{c}(:,m);
            Kc(n,m) = kernel_func(xn,xm,HP);
        end
    end
    Ni{c} = Kc * (eye(li{c}) - (1/li{c})*ones(li{c})) * Kc';
    % Calculate M and N Matrices
    M = M + li{c} * (Mi{c} - Mst) * (Mi{c} - Mst)';
    N = N + Ni{c};
end
N = N + sig2n*eye(L);

% Get eigenvectors and eigenvalues of Kernel matrices

[Veig,Leig] = eig(N\M);
Veig = abs(Veig);
Leig = abs(Leig);
Leig = diag(Leig);

% Sort eigenvalues and eigenvectors

SORT = bubble_sort(Leig,2);	% uses eigenvalues to define the order
Leig = Leig(SORT.ind);      % sort eigenvalues
Veig = Veig(:,SORT.ind); 	% sort eigenvectors

% Explained variance

Ev = zeros(1,L);
Ev(1) = Leig(1);
for i = 2:L,
    Ev(i) = Ev(i-1) + Leig(i);
end
Ev = Ev/sum(Leig);

% Find number of Principal Components

for i = 1:L,
    if(Ev(i) >= tol || i == p),
        q = i;
        break;
    end
end

% Get transformation matrix [p x q]

A = Veig(:,1:q);

% Transform input matrix

Xtr = zeros(q,L);

for n = 1:L,
    xn = X(:,n);
    for k = 1:q,
        ak = A(:,k);
        for i = 1:L,
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
PAR.L = Leig;
PAR.V = Veig;
PAR.W = A;
PAR.q = q;
PAR.Ev = Ev;

%% END