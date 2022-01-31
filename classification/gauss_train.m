function [PARout] = gauss_train(DATA,PAR)

% --- Gaussian Classifier Training ---
%
%   [PARout] = gauss_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix                           [p x N]
%           output = labels matrix                              [Nc x N]
%       PAR. 
%           type = type of gaussian classifier                  [cte]
%               1: gi(x) = Qi(x) + ln(det(Ci)) - 2ln(p(Ci))
%               2: gi(x) = Qi(x) + ln(det(Ci))
%               3: gi(x) = Qi(x) (mahalanobis distance)
%            	        (covariance matrix is the pooled covariance matrix)
%               4: gi(x) = ||x-mi||^2 (euclidean distance)
%               5: gi(x) = naive bayes (uncorrelated data)
%   Output:
%       PARout.
%           Ni = number of "a priori samples" per class         [Nc x 1]
%           mu_i = centroid of each class                       [Nc x p]
%           Ci = covariance matrix of each class                [Nc x p x p]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.type = 2;          % Type of classificer
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'type')))
        PAR.type = 2;
    end
end

%% INITIALIZATIONS

% Data matrix
X = DATA.input;
Y = DATA.output;

% Problem Init
[Nc,~] = size(Y);   % Number or classes
[p,N] = size(X);    % Number of attributes and samples

% Input matrix with [N x p] dimension
X = X';

% Output Matrix with [N x Nc] dimension
Y = Y';
Y_aux = zeros(N,1);
for i = 1:N
    Y_aux(i) = find(Y(i,:) > 0);
end
Y = Y_aux;

%% ALGORITHM

Ni = zeros(Nc,1);	% Number of samples per class
mu_i = zeros(Nc,p);	% Mean of samples for each class

for i = 1:N
    mu_i(Y(i),:) = mu_i(Y(i),:) + X(i,:);	% sample sum accumulator
    Ni(Y(i)) = Ni(Y(i)) + 1;                % number of samples accumulator
end

for i = 1:Nc
    mu_i(i,:) = mu_i(i,:)/Ni(i);            % calculate mean
end

% Calculte Covariance Matrix for each class [(X-mu)*(X-mu)']/N

% Initialize matrix
Ci = cell(1,Nc);
for i = 1:Nc
    Ci{i} = zeros(p,p);
end

% Calculate iteratively
for i = 1:N
    Ci{Y(i)} = Ci{Y(i)} + (X(i,:)-mu_i(Y(i),:))'*(X(i,:)-mu_i(Y(i),:));
end

% Divide by number of elements
for i = 1:Nc
    Ci{i} = Ci{i}/Ni(i);
    % Decorrelates data (for naive bayes)
    if PAR.type == 5
        Ci{i} = diag(diag(Ci{i}));
    end
    Ci{i} = Ci{i} + 0.0001*eye(p,p);
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.Ni = Ni;
PARout.mu_i = mu_i;
PARout.Ci = Ci;

%% END