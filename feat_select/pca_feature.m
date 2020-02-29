function [PAR] = pca_feature(DATA,HP)

% --- Principal Component Analysis for Feature Selection ---
%
%   PAR = pca_feature(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                            [p x N]
%           output = output matrix                          [Nc x N]
%       HP.
%           tol = tolerance of explained value  [0 - 1]     [cte]
%           rem = mean removal [0 or 1]                     [cte]
%   Output:
%       PAR.
%           mu = mean of input matrix                       [p x 1]
%           q = number of used attributes                   [cte]
%           input = attributes matrix                       [q x N]
%           output = labels matrix                          [Nc x N]
%           L = eigenvalues of cov                          [1 x p]
%           V = eigenvectors of cov                         [p x p]
%           W = Transformation Matrix                       [p x q]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    HPaux.tol = 1;
    HPaux.rem = 1;
	HP = HPaux;
else
    if (~(isfield(HP,'tol'))),
        HP.tol = 1;
    end
    if (~(isfield(HP,'rem'))),
        HP.rem = 1;
    end
end

%% INITIALIZATIONS

% Get Data
X = DATA.input;     % input matrix
Y = DATA.output;    % output matrix
[p,N] = size(X);    % dimensions of input matrix

% Get Hyperparameters
tol = HP.tol;   	% explained value
rem = HP.rem;       % remove or not mean

%% ALGORITHM

% Calculate Mean and Remove it from each sample 

Xmean = mean(X,2);
if (rem == 1),
    X = X - repmat(Xmean,1,N);
end

% Calculate Covariance Matrix

Cx = cov(X');

% Get eigenvectors and eigenvalues of covariance matrix

[V,L] = eig(Cx);
L = diag(L);

% Sort eigenvalues and eigenvectors

SORT = bubble_sort(L,2);    % uses eigenvalues to define the order
L = L(SORT.ind);            % sort eigenvalues
V = V(:,SORT.ind);          % sort eigenvectors

% Explained variance

Ev = zeros(1,p);
Ev(1) = L(1);
for i = 2:p,
    Ev(i) = Ev(i-1) + L(i);
end
Ev = Ev/sum(L);

% Find number of Principal Components

for i = 1:p,
    if(Ev(i) >= tol),
        q = i;
        break;
    end
end

% Get transformation matrix [p x q]

W = V(:,1:q);

% Transform input matrix

X = W' * X;     % generate matrix -> [q x N]

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.mu = Xmean;
PAR.input = X;
PAR.output = Y;
PAR.L = L;
PAR.V = V;
PAR.W = W;
PAR.q = q;

%% END