function [PARout] = kmeans_train(DATA,PAR)

% --- k-means based classifier training ---
%
%   [PARout] = kmeans_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                            [p x Ntr]
%           output = output matrix                          [Nc x Ntr]
%       PAR.
%           Nep = max number of epochs                   	[cte]
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
%           lbl = type of labeling                          [cte]
%               1: Majority voting
%               2: Average distance
%               3: Minimum distance
%           Von = enable or disable video                   [cte]
%           Ktype = Kernel Type                             [cte]
%               = 0 -> non-kernelized algorithm
%           K = number of nearest neighbors                 [cte]
%   Output:
%       PARout.
%       	Cx = clusters centroids (prototypes)            [p x Nk]
%           Cy = class of each prototype/neuron             [Nc x Nk]
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

%% ALGORITHM

OUT_CL = kmeans_cluster(DATA,PAR);
PARout = kmeans_label(DATA,OUT_CL);

%% END