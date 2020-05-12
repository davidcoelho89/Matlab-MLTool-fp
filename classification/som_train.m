function [PARout] = som_train(DATA,PAR)

% --- SOM based classifier training ---
%
%   [PARout] = som_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                            [p x N]
%           output = output matrix                          [Nc x N]
%       PAR.
%           Nep = max number of epochs                    	[cte]
%           Nk = number of prototypes (neurons)             [1 x Nd]
%                (Nd = dimenions)
%           init = type of initialization for prototypes    [cte]
%               1: C = zeros
%               2: C = randomly picked from data
%               3: C = mean of randomly choosen data
%               4: C = between max and min values of atrib
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
%           Nn = number of neighbors (for training)         [cte]
%           neig = type of neighborhood function            [cte]
%               1: if winner, h = 1, else h = 0.
%               2: if neighbor, h = exp (-(||ri -ri*||^2)/(V^2))
%               	where: V = Vo*((Vt/Vo)^(t/tmax))
%               3: Decreasing function 1. Init with 3 neig
%           Vo = initial neighborhood parameter             [cte]
%           Vt = final neighborhood parameter               [cte]
%           lbl = type of labeling                          [cte]
%               1: Majoriting voting
%               2: Average distance
%               3: Minimum distance
%           Von = enable or disable video                   [cte]
%           K = number of nearest neighbors                 [cte]
%           Ktype = Kernel Type                             [cte]
%               = 0 -> non-kernelized algorithm
%   Output:
%       PARout.
%       	Cx = clusters centroids (prototypes)            [p x Nk]
%           Cy = class of each prototype/neuron             [Nc x Nk]
%           ind = cluster index for each sample             [Nd x Ntr]
%           SSE = Sum of Squared Errors for each epoch      [1 x Nep]
%           VID = frame struct (played by 'video function')	[1 x Nep]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nep = 200;     	% max number of epochs
    PARaux.Nk = [4 3];    	% number of neurons (prototypes)
    PARaux.init = 02;     	% neurons' initialization
    PARaux.dist = 02;      	% type of distance
    PARaux.learn = 02;     	% type of learning step
    PARaux.No = 0.7;       	% initial learning step
    PARaux.Nt = 0.01;      	% final learnin step
    PARaux.Nn = 01;      	% number of neighbors
    PARaux.neig = 02;      	% type of neighborhood function
    PARaux.Vo = 0.8;      	% initial neighborhood constant
    PARaux.Vt = 0.3;      	% final neighborhood constant
    PARaux.lbl = 1;         % Neurons' labeling function
    PARaux.Von = 0;         % disable video 
    PARaux.K = 1;           % nearest neighbor scheme
    PARaux.Ktype = 0;       % Non-kernelized Algorithm
    PAR = PARaux;
else
    if (~(isfield(PAR,'Nep'))),
        PAR.Nep = 200;
    end
    if (~(isfield(PAR,'Nk'))),
        PAR.Nk = [4 3];
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
    if (~(isfield(PAR,'Nn'))),
        PAR.Nn = 1;
    end
    if (~(isfield(PAR,'neig'))),
        PAR.neig = 2;
    end
    if (~(isfield(PAR,'Vo'))),
        PAR.Vo = 0.8;
    end
    if (~(isfield(PAR,'Vt'))),
        PAR.Vt = 0.3;
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

[OUT_CL] = som_cluster(DATA,PAR);
[PARout] = som_label(DATA,OUT_CL);

%% END