function [PARout] = ng_cluster(DATA,PAR)

% --- NG Clustering Function ---
%
%   [PARout] = ng_cluster(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                            [p x Ntr]
%       PAR.
%           Nep = max number of epochs                   	[cte]
%           Nk = number of clusters / prototypes            [cte]
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
%           ind = cluster index for each sample             [1 x Ntr]
%           SSE = Sum of Squared Errors for each epoch      [1 x Nep]
%           VID = frame struct (played by 'video function')	[1 x Nep]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nep = 200;       % max number of epochs
    PARaux.Nk = 12;         % number of neurons (prototypes)
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

%% INITIALIZATION

% Get Data

X = DATA.input;
[~,N] = size(X);

% Get hyperparameters

Nep = PAR.Nep;
Nk = PAR.Nk;
learn = PAR.learn;
No = PAR.No;
Nt = PAR.Nt;
Von = PAR.Von;

% Init aux variables

tmax = N*Nep;       % max number of iterations

% Init Outputs

if (isfield(PAR,'Cx')),
    C = PAR.Cx;
    [~,Nk] = size(C);
else
    C = prototypes_init(DATA,PAR);
end

ind = zeros(1,N);

SSE = zeros(1,Nep);

VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep,
    
	% Save frame of the current epoch
    if (Von),
        VID(ep) = prototypes_frame(C,DATA);
    end

    % shuffle data
    I = randperm(N);
    X = X(:,I);
    
    % Update Prototypes (one epoch)
    for t = 1:N,

        % Get Sample
        xt = X(:,t);                        

        % Get Learning Step
        n = prototypes_learn(learn,tmax,t,No,Nt);	
        
        % Get neighborhood vector
        h = ng_f_neig(C,xt,PAR,t,tmax);
       
        % Uptade Prototypes - 1D
        for prot = 1:Nk,
            C(:,prot) = C(:,prot) + n*h(prot)*( xt - C(:,prot) );
        end
        
    end
    
    % SSE (one epoch)
    SSE(ep) = prototypes_sse(C,DATA,PAR);
    
end

% Assign indexes
for t = 1:N,
    xt = DATA.input(:,t);          	% not shuffled data
    win = prototypes_win(C,xt,PAR);	% Winner Neuron index
    ind(:,t) = win;                	% save index for sample
end

%% FILL OUTPUT STRUCTURE

% Get Previous Parameters
PARout = PAR;

% Get Output Parameters
PARout.Cx = C;
PARout.ind = ind;
PARout.SSE = SSE;
PARout.VID = VID;

%% END