function [PARout] = lvq_cluster(DATA,PAR)

% --- LVQ Clustering Function ---
%
%   [PARout] = lvq_cluster(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                            [p x Ntr]
%           output = output matrix                          [Nc x Ntr]
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
    PARaux.Nep = 200;    	% max number of epochs
    PARaux.Nk = 20;      	% number of neurons (prototypes)
    PARaux.init = 02;   	% neurons' initialization
    PARaux.dist = 02;      	% type of distance
    PARaux.learn = 02;    	% type of learning step
    PARaux.No = 0.7;      	% initial learning step
    PARaux.Nt = 0.01;      	% final   learning step
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
    if (~(isfield(PAR,'learn'))),
        PAR.learn = 2;
    end
    if (~(isfield(PAR,'No'))),
        PAR.No = 0.7;
    end
    if (~(isfield(PAR,'Nt'))),
        PAR.Nt = 0.01;
    end
    if (~(isfield(PAR,'Von'))),
        PAR.Von = 0;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 0;
    end
    if (~(isfield(PAR,'k'))),
        PAR.K = 1;
    end
end

%% INITIALIZATION

% Get data 

X = DATA.input;
Y = DATA.output;
[~,N] = size(X);

% Get parameters

Nep = PAR.Nep;
learn = PAR.learn;
No = PAR.No;
Nt = PAR.Nt;
Von = PAR.Von;

% Init aux variables

tmax = N*Nep;           % max number of iterations
t = 0;                 	% count iterations

% Init Outputs

if (isfield(PAR,'Cx')),
    Cx = PAR.Cx;
    Cy = PAR.Cy;
else
    C = lvq_f_init(DATA,PAR);
    Cx = C.x;
    Cy = C.y;
end

ind = zeros(1,N);

SSE = zeros(1,Nep);

VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

% Update prototypes (just winner for each iteration)
for ep = 1:Nep,
    
    % Save frame of the current epoch
    if (Von),
        VID(ep) = prototypes_frame(Cx,DATA);
    end
    
    % shuffle data
    I = randperm(N);
    X = X(:,I);
    Y = Y(:,I);
    
    % Get Winner Neuron, update Learning Step, update prototypes
    for i = 1:N,

        t = t+1;                                    % Uptade Iteration
        sample = X(:,i);                            % Training sample
        win = prototypes_win(Cx,sample,PAR);     	% Winner Neuron index
        n = prototypes_learn(learn,tmax,t,No,Nt);   % Learning Step
        
        % Update Winner Neuron (Prototype)
        [~,y] = max(Y(:,i));
        [~,yh] = max(Cy(:,win));
        if (yh == y),
            Cx(:,win) = Cx(:,win) +  n * (X(:,i) - Cx(:,win));
        else
            Cx(:,win) = Cx(:,win) -  n * (X(:,i) - Cx(:,win));
        end
    end
    
    % SSE (one epoch)
    SSE(ep) = prototypes_sse(Cx,DATA,PAR);
    
end

% Assign indexes
for i = 1:N,
    sample = DATA.input(:,i);               % not shuffled data
    win = prototypes_win(Cx,sample,PAR);  	% Winner Neuron index
    ind(:,i) = win;                         % save index for sample
end

%% FILL OUTPUT STRUCTURE

% Get Previous Parameters
PARout = PAR;

% Get Output Parameters
PARout.Cx = Cx;
PARout.Cy = Cy;
PARout.ind = ind;
PARout.SSE = SSE;
PARout.VID = VID;

%% END