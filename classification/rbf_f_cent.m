function [C] = rbf_f_cent(DATA,PAR)

% --- Initialize centroids for RBF ---
%
%   [C] = rbf_f_init(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%       PAR.
%           Nh = number of hidden neurons (centroids)           [cte]
%           init = how will be the prototypes' initial values   [cte]
%               1: Forgy Method (randomly choose k observations)
%               2: Vector Quantization (kmeans)
%   Output:
%       C = prototypes matrix                                   [p x Nh]

%% INITIALIZATIONS

% Get Data
X = DATA.input;
[p,N] = size(X);

% Get Parameters
init = PAR.init;
Nk = PAR.Nh;

% Initialize prototypes matrix
C = zeros(p,Nk);

%% ALGORITHM

if (init == 1)
    % Randomly choose one
    I = randperm(N);
    C = X(:,I(1:Nk));

elseif (init == 2)
    % Vector Quantization (kmeans)
    HP.Nk = Nk;
    OUT = kmeans_cluster(DATA,HP);
    C = OUT.Cx;
else
    disp('Unknown initialization. Prototypes = 0.');
    
end

%% END