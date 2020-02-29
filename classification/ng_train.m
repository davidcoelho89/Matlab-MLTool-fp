function [PARout] = ng_train(DATA,PAR)

% --- NG based classifier training ---
%
%   [PARout] = ng_train(DATA,PAR)
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
%           learn = type of learning step                   [cte]
%               1: N = No (constant)
%               2: N = No*(1-(t/tmax))
%               3: N = No/(1+t)
%               4: N = No*((Nt/No)^(t/tmax))
%           No = initial learning step                      [cte]
%           Nt = final learning step                        [cte]
%           neig = type of neighborhood function            [cte]
%               1: L = Lo (constant)
%               2: L = Lo*((Lt/Lo)^(t/tmax))
%           Lo = initial neighborhood parameter             [cte]
%           Lt = final neighborhood parameter               [cte]
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
    PARaux.Nep = 200;   	% max number of epochs
    PARaux.Nk = 12;       	% number of neurons (prototypes)
    PARaux.init = 02;   	% neurons' initialization
    PARaux.dist = 02;      	% type of distance
    PARaux.learn = 02;    	% type of learning step
    PARaux.No = 0.7;      	% initial learning step
    PARaux.Nt = 0.01;      	% final   learning step
    PARaux.neig = 02;      	% type of neighborhood function
    PARaux.Lo = 0.8;      	% initial neighborhood constant
    PARaux.Lt = 0.3;      	% final neighborhood constant
    PARaux.lbl = 1;         % Prototypes' labeling function
    PARaux.Von = 0;         % disable video
    PARaux.Ktype = 0;       % Non-kernelized Algorithm
    PARaux.K = 1;           % Number of nearest neighbors
    PAR = PARaux;
else
    if (~(isfield(PAR,'Nep'))),
        PAR.Nep = 200;
    end
    if (~(isfield(PAR,'Nk'))),
        PAR.Nk = 12;
    end
    if (~(isfield(PAR,'init'))),
        PAR.init = 2;
    end
    if (~(isfield(PAR,'dist'))),
        PAR.dist = 2;
    end
    if (~(isfield(PAR,'learn'))),
        PAR.learn = 2;
    end
    if (~(isfield(PAR,'No'))),
        PAR.No = 0.7;
    end
    if (~(isfield(PAR,'Nt'))),
        PAR.Nt = 0.01;
    end
    if (~(isfield(PAR,'neig'))),
        PAR.neig = 2;
    end
    if (~(isfield(PAR,'Lo'))),
        PAR.Lo = 0.8;
    end
    if (~(isfield(PAR,'Lt'))),
        PAR.Lt = 0.3;
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

[OUT_CL] = ng_cluster(DATA,PAR);
[PARout] = ng_label(DATA,OUT_CL);

%% END