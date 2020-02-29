function [C] = lvq_f_init(DATA,PAR)

% --- Initialize prototypes for LVQ clustering ---
%
%   [C] = lvq_f_init(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                            [p x Ntr]
%           output = output matrix                          [Nc x Ntr]
%       PAR.
%           Nk = number of prototypes
%           init = how will be the prototypes' initial values
%               1: C = zeros (mean of normalized data)
%               2: Forgy Method (randomly choose k observations from 
%                  data set)
%               3: Randomly assign a cluster to each observation, than
%                  update clusters' centers
%               4: prototype's values randomly choosed between min and
%                  max values of data's atributtes.
%               5: Clustering algorithm for each class.
%   Output:
%       C.
%           x = clusters centroids (prototypes)             [p x Nk]
%           y = class of each prototype/neuron              [Nc x Nk]

%% INITIALIZATIONS

% Get Data

X = DATA.input;
[p,N] = size(X);
Y = DATA.output;
[Nc,~] = size(Y);

% Get Parameters

init = PAR.init;
Nk = PAR.Nk;

% Init outputs

Cx = zeros(p,Nk);
Cy = -1*ones(Nc,Nk);

%% ALGORITHM
   

if (init == 1)
    % Calculates prototypes per class
    ppc = floor(Nk/Nc);
    % init counter
    cont = 0;
    % assign labels to prototypes 
    for i = 1:Nc,
        if i ~= Nc,
            Cy(i,cont+1:cont+ppc) = 1;
        else
            Cy(i,cont+1:end) = 1;
        end
        cont = cont + ppc;
    end
    
elseif (init == 2)
    I = randperm(N);
    Cx = X(:,I(1:Nk));
    Cy = Y(:,I(1:Nk));
    
elseif (init == 3)
    % Initialize number of samples for each cluster
    n_samples = zeros(1,Nk);
    % initialize randomly each sample index
    I = rand(1,N);
    index = ceil(Nk*I);
    % calculate centroids
    for i = 1:N,
        n_samples(index(i)) = n_samples(index(i)) + 1;
        Cx(:,index(i)) = Cx(:,index(i)) + X(:,i);
    end
    for i = 1:Nk,
        Cx(:,i) = Cx(:,i) / n_samples(i);
    end
    
    % Calculates prototypes per class
    ppc = floor(Nk/Nc);
    % init counter
    cont = 0;
    % assign labels to prototypes 
    for i = 1:Nc,
        if i ~= Nc,
            Cy(i,cont+1:cont+ppc) = 1;
        else
            Cy(i,cont+1:end) = 1;
        end
        cont = cont + ppc;
    end

elseif (init == 4),
    % Calculate min and max value of parameters
    [pmin,~] = min(X,[],2);
    [pmax,~] = max(X,[],2);
    % generate vectors
    for i = 1:Nk,
        Cx(:,i) = pmin + (pmax - pmin).*rand(p,1);
    end 
    
    % Calculates prototypes per class
    ppc = floor(Nk/Nc);
    % init counter
    cont = 0;
    % assign labels to prototypes 
    for i = 1:Nc,
        if i ~= Nc,
            Cy(i,cont+1:cont+ppc) = 1;
        else
            Cy(i,cont+1:end) = 1;
        end
        cont = cont + ppc;
    end
else
    disp('Unknown initialization. Prototypes = 0.');
end

%% FILL OUTPUT STRUCTURE

C.x = Cx;
C.y = Cy;

%% END