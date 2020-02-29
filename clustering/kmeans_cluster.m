function [PARout] = kmeans_cluster(DATA,PAR)

% --- k-means clustering function ---
%
%   [PARout] = kmeans_cluster(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                            [p x Ntr]
%       PAR.
%           Nep = max number of epochs                      [cte]
%           Nk = number of clusters / prototypes            [cte]
%           init = type of initialization for prototypes    [cte]
%               1: Cx = zeros
%               2: Cx = randomly picked from data
%               3: Cx = mean of randomly choosen data
%               4: Cx = between max and min values of atrib
%           dist = type of distance                         [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           Von = enable or disable video                   [cte]
%           Ktype = Kernel Type                             [cte]
%               = 0 -> non-kernelized algorithm
%           K = number of nearest neighbors                 [cte]
%   Output:
%       PARout.
%       	Cx = clusters prototypes                      	[p x Nk]
%           ind = cluster index for each sample             [1 x Ntr]
%           SSE = Sum of Squared Errors for each epoch      [1 x Nep]
%           VID = frame struct (played by 'video function')	[1 x Nep]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nep = 200;       % max number of epochs
    PARaux.Nk = 20;         % number of clusters (prototypes)
    PARaux.init = 2;        % type of initialization
    PARaux.dist = 2;        % type of distance
    PARaux.lbl = 1;         % Neurons' labeling function
    PARaux.Von = 0;         % disable video
    PARaux.Ktype = 0;       % Non-kernelized Algorithm
    PARaux.K = 1;           % Number of nearest neighbors
    PAR = PARaux;
else
    if (~(isfield(PAR,'Nep'))),
        PAR.Nep = 200;
    end
    if (~(isfield(PAR,'Nk'))),
        PAR.Nk = 20;
    end
    if (~(isfield(PAR,'init'))),
        PAR.init = 2;
    end
    if (~(isfield(PAR,'dist'))),
        PAR.dist = 2;
    end
    if (~(isfield(PAR,'lbl'))),
        PAR.lbl = 1;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 0;
    end
    if (~(isfield(PAR,'K'))),
        PAR.K = 1;
    end
end

%% INITIALIZATIONS

% Get Data

X = DATA.input;
[p,N] = size(X);

% Get Hyperparameters

Nep = PAR.Nep;
Nk = PAR.Nk;
Von = PAR.Von;

% Init output variables

if (isfield(PAR,'Cx'))
    Cx = PAR.Cx;
    [~,Nk] = size(Cx);
else
    Cx = prototypes_init(DATA,PAR);
end
ind = zeros(1,N);
SSE = zeros(1,Nep);

VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep,

    % Save frame of the current epoch
    
    if (Von),
        VID(ep) = prototypes_frame(Cx,DATA);
    end

    % Flag to verify if there was any change of data labeling
    
    change_flag = 0;

    % Assign data points to each cluster
    
    for i = 1:N,

        % Calculate closer centroid to sample
        x = X(:,i);
        current_index = prototypes_win(Cx,x,PAR);

        % Update index and signs that a change occured
        if (ind(i) ~= current_index),
            ind(i) = current_index;
            change_flag = 1;
        end

    end

    % Calculate SSE for the current loop
    
    SSE(ep) = prototypes_sse(Cx,DATA,PAR);

    % If there wasn't any change, break the loop
    
    if (change_flag == 0),
        SSE = SSE(1:ep);
        break;
    end

    % Calculate new centroids
    
    n_samples = zeros(1,Nk);    % init number of samples per centroid
    Cx = zeros(p,Nk);            % init parameters of centroids
    
    for i = 1:N,
        n_samples(ind(i)) = n_samples(ind(i)) + 1;
        Cx(:,ind(i)) = Cx(:,ind(i)) + X(:,i);
    end
    
    for i = 1:Nk,
        % Calculate mean of samples belonging to a cluster
        if n_samples(i) ~= 0
            Cx(:,i) = Cx(:,i) / n_samples(i);
        % Randomly assigns a sample to a cluster centroid
        else
            I = randperm(N);
            Cx(:,i) = X(:,I(i));
        end
    end

end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.Cx = Cx;
PARout.ind = ind;
PARout.SSE = SSE(1:ep);
PARout.VID = VID(1:ep);

%% END