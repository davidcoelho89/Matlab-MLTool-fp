function [PAR] = lda_feature(DATA,HP)

% --- Linear Discriminant Analysis for Feature Selection ---
%
%   PAR = lda_feature(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       HP.
%           tol = tolerance of explained value  [0 - 1]         [cte]
%           rem = mean removal [0 or 1]                         [cte]
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
[Nc,~] = size(Y);   % number of classes

% Get Hyperparameters
tol = HP.tol;   	% explained value
rem = HP.rem;       % remove or not mean
sig2n = 0.001;      % Regularization factor

%% ALGORITHM

% Calculate Mean and Remove it from each sample 

m = mean(X,2);
if (rem == 1),
    X = X - repmat(m,1,N);
end

% Calculate Scatter Matrices (Sw and Sb)

Xi = cell(Nc,1);
mi = cell(Nc,1);
Ni = cell(Nc,1);
Pi = cell(Nc,1);
Si = cell(Nc,1);

Sw = zeros(p,p);
Sb = zeros(p,p);

[~,Y_seq] = max(Y);

for j = 1:Nc,
    Xi{j} = X(:,(Y_seq == j));
    mi{j} = mean(Xi{j},2);
    Ni{j} = length(find(Y_seq == j));
    Mi = repmat(mi{j},1,Ni{j});
    Pi{j} = Ni{j} / N;
    Si{j} = (Xi{j} - Mi) * (Xi{j} - Mi)' / Ni{j};
    Sw = Sw + Pi{j} * Si{j};
    Sb = Sb + Pi{j} * (mi{j} - m) * (mi{j} - m)';
end

Sw = Sw + sig2n*eye(p); % Improve Conditioning number of Sw

% Get eigenvectors and eigenvalues of scatter matrices

[V,L] = eig(Sw\Sb);
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
PAR.mu = m;
PAR.input = X;
PAR.output = Y;
PAR.L = L;
PAR.V = V;
PAR.W = W;
PAR.q = q;

%% END