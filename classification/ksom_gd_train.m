function [PARout] = ksom_gd_train(DATA,PAR)

% --- KSOM-GD Training Function ---
%
%   PARout = ksom_gd_train(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       PAR.
%           Nep = max number of epochs                         	[cte]
%           Nk = number of prototypes (neurons)                 [1 x Nd]
%                (Nd = dimenions)
%           init = type of initialization for prototypes        [cte]
%               1: Cx = zeros
%               2: Cx = randomly picked from data
%               3: Cx = mean of randomly choosen data
%               4: Cx = between max and min values of atrib
%           dist = type of distance                             [cte]
%               0:    Dot product
%               inf:  Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1:    Manhattam (city-block) distance
%               2:    Euclidean distance
%               >2:   Minkowsky distance
%           learn = type of learning step                       [cte]
%               1: N = No (constant)
%               2: N = No*(1-(t/tmax))
%               3: N = No/(1+t)
%               4: N = No*((Nt/No)^(t/tmax))
%           No = initial learning step                          [cte]
%           Nt = final learning step                            [cte]
%           Nn = number of neighbors                            [cte]
%           neig = type of neighborhood function                [cte]
%               1: if winner, h = 1, else h = 0.
%               2: if neighbor, h = exp (-(||ri -ri*||^2)/(V^2))
%                   where: V = Vo*((Vt/Vo)^(t/tmax))
%               3: Decreasing function 1. Init with 3 neig
%           Vo = initial neighborhood parameter                 [cte]
%           Vt = final neighborhood parameter                   [cte]
%           lbl = type of labeling                              [cte]
%               1: Majoriting voting
%               2: Average distance
%               3: Minimum distance
%           Von = enable or disable video                       [cte]
%           K = Number of nearest neighbors (classify)          [cte]
%           knn_type = Type of knn aproximation                 [cte]
%               1: Majority Voting
%               2: Weighted KNN
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PARout.
%       	Cx = clusters centroids (prototypes)            [p x Nk]
%           Cy = class of each prototype/neuron             [Nc x Nk]
%           R = prototypes' grid positions                  [Nd x Nk]
%           ind = cluster index for each sample             [Nd x Ntr]
%           SSE = Sum of Squared Errors for each epoch      [1 x Nep]
%           VID = frame struct (played by 'video function')	[1 x Nep]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.Nep = 200;       % max number of epochs
    PARaux.Nk = [5 4];    	% number of neurons (prototypes)
    PARaux.init = 2;        % neurons' initialization
    PARaux.dist = 2;        % type of distance
    PARaux.learn = 2;       % type of learning step
    PARaux.No = 0.7;        % initial learning step
    PARaux.Nt = 0.01;       % final learning step
    PARaux.Nn = 1;          % number of neighbors
    PARaux.neig = 2;        % type of neighbor function
    PARaux.Vo = 0.8;        % initial neighbor constant
    PARaux.Vt = 0.3;        % final neighbor constant
    PARaux.lbl = 1;         % Neurons' labeling function
    PARaux.Von = 0;         % disable video
    PARaux.K = 1;           % nearest neighbor scheme
    PARaux.knn_type = 1;    % Majority voting for knn
    PARaux.Ktype = 2;       % Gaussian Kernel
    PARaux.sigma = 2;       % Kernel standard deviation (gaussian)
    PAR = PARaux;
else
    if (~(isfield(PAR,'Nep')))
        PAR.Nep = 200;
    end
    if (~(isfield(PAR,'Nk')))
        PAR.Nk = [5 4];
    end
    if (~(isfield(PAR,'init')))
        PAR.init = 2;
    end
    if (~(isfield(PAR,'dist')))
        PAR.dist = 2;
    end
    if (~(isfield(PAR,'learn')))
        PAR.learn = 2;
    end
    if (~(isfield(PAR,'No')))
        PAR.No = 0.7;
    end
    if (~(isfield(PAR,'Nt')))
        PAR.Nt = 0.01;
    end
    if (~(isfield(PAR,'Nn')))
        PAR.Nn = 1;
    end
    if (~(isfield(PAR,'neig')))
        PAR.neig = 2;
    end
    if (~(isfield(PAR,'Vo')))
        PAR.Vo = 0.8;
    end
    if (~(isfield(PAR,'Vt')))
        PAR.Vt = 0.3;
    end
    if (~(isfield(PAR,'lbl')))
        PAR.lbl = 1;
    end
    if (~(isfield(PAR,'Von')))
        PAR.Von = 0;
    end
    if (~(isfield(PAR,'K')))
        PAR.K = 1;
    end
    if (~(isfield(PAR,'knn_type')))
        PAR.knn_type = 1;
    end
    if (~(isfield(PAR,'Ktype')))
        PAR.Ktype = 2;
    end
    if (~(isfield(PAR,'sigma')))
        PAR.sigma = 2;
    end
end

%% ALGORITHM

OUT_CL = ksom_gd_cluster(DATA,PAR);
PARout = ksom_gd_label(DATA,OUT_CL);

%% END