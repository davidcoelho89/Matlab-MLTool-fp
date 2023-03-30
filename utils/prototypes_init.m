function [Cx] = prototypes_init(DATA,PAR)

% --- Initialize prototypes for clustering ---
%
%   [C] = prototypes_init(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%       PAR.
%           Nk = number of prototypes                           [1 x Nd]
%                (Nd = dimenions)
%           init = how will be the prototypes' initial values
%               1: C = zeros (mean of normalized data)
%               2: Forgy Method (randomly choose k observations from data set)
%               3: Randomly assign a cluster to each observation, than
%                  update clusters' centers
%               4: prototype's values randomly choosed between min and
%                  max values of data's atributtes.
%   Output:
%       Cx = prototypes matrix [p x Nk]

%% INITIALIZATIONS

% Get Data

X = DATA.input;
[p,N] = size(X);

% Get Parameters

init = PAR.init;
Nk = prod(PAR.Nk);

% Init Outputs

Cx = zeros(p,Nk);

%% ALGORITHM

if (init == 1) 

    % Does nothing: already initialized with zeros

elseif (init == 2)

    I = randperm(N);
    Cx = X(:,I(1:Nk));

elseif (init == 3)

    % Initialize number of samples for each cluster
    n_samples = zeros(1,Nk);
    % initialize randomly each sample index
    I = rand(1,N);
    index = ceil(Nk*I);
    % calculate centroids
    for i = 1:N
        n_samples(index(i)) = n_samples(index(i)) + 1;
        Cx(:,index(i)) = Cx(:,index(i)) + X(:,i);
    end
    for i = 1:Nk
        Cx(:,i) = Cx(:,i) / n_samples(i);
    end

elseif (init == 4)

    % Calculate min and max value of parameters
    [pmin,~] = min(X,[],2);
    [pmax,~] = max(X,[],2);
    % generate vectors
    for i = 1:Nk
        Cx(:,i) = pmin + (pmax - pmin).*rand(p,1);
    end

else
    disp('Unknown initialization. Prototypes = 0.');

end

%% END