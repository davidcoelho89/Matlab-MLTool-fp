function [PARout] = ksom_ps_cluster(DATA,PAR)

% --- KSOM-PS Clustering Function ---
%
%   [PARout] = ksom_ps_cluster(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%       PAR.
%           Nep = max number of epochs                         	[cte]
%           Nk = number of prototypes (neurons)                 [1 x Nd]
%                (Nd = dimenions)
%           init = type of initialization for prototypes        [cte]
%               1: C = zeros
%               2: C = randomly picked from data
%               3: C = mean of randomly choosen data
%               4: C = between max and min values of atrib
%           dist = type of distance                             [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance  
%               2: Euclidean distance
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
%           Ss = Selection strategy                             [cte]
%               1: Randomly
%               2: Renyi's entropy 
%               3: ALD
%               4: Coherence
%           v1 = Sparseness parameter 1                         [cte]
%           M = no of samples used to estimate kernel matrix    [cte]
%           K = number of nearest neighbors                     [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PARout.
%       	Cx = clusters centroids (prototypes) [p x Nk(1) x ... x Nk(Nd)]
%           ind = cluster index for each sample             [Nd x Ntr]
%           SSE = Sum of Squared Errors for each epoch      [1 x Nep]
%           VID = frame struct (played by 'video function')	[1 x Nep]
%           Mred = matrix with training data samples        [p x m]
%                  used to estimate the feature space
%           eig_vec = eigenvectors of Mred                  [m x m]
%           eig_val = eigenvalues of Mred                   [1 x m]
%           Mphi = mapped input matrix                      [m x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
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
    PARaux.Ss = 1;        	% Selection strategy
    PARaux.v1 = 0.01;       % Sparseness parameter 1
    PARaux.M = 50;          % no of samples used to estimate kernel matrix
    PARaux.K = 1;           % nearest neighbor scheme
    PARaux.Ktype = 2;       % Gaussian Kernel
    PARaux.sigma = 2;       % Kernel standard deviation (gaussian)
    PAR = PARaux;
else
    if (~(isfield(PAR,'Nep'))),
        PAR.Nep = 200;
    end
    if (~(isfield(PAR,'Nk'))),
        PAR.Nk = [5 4];
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
    if (~(isfield(PAR,'Ss'))),
        PAR.Ss = 1;
    end
    if (~(isfield(PAR,'v1'))),
        PAR.v1 = 0.01;
    end
    if (~(isfield(PAR,'M'))),
        PAR.M = 50;
    end
    if (~(isfield(PAR,'K'))),
        PAR.K = 1;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 2;
    end
    if (~(isfield(PAR,'sigma'))),
        PAR.sigma = 2;
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
Nn = PAR.Nn;
neig = PAR.neig;
Vo = PAR.Vo;
Vt = PAR.Vt;
Von = PAR.Von;
Ktype = PAR.Ktype;

% Init aux variables

tmax = N*Nep;       % max number of iterations
t = 0;              % count iterations

% Init outputs

ind = zeros(2,N);   % ToDo - init as SOM

SSE = zeros(1,Nep);

VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

% Select the prototypes from training data

Mred = prototypes_select(DATA,PAR);
[~,M] = size(Mred);

% Calculate Kernel Matrix form Reduced Matrix [p x m]

Kmat = kernel_mat(Mred,PAR);

% Eigendecomposition of Kernel Matrix

[U_aux,D_aux] = eig(Kmat);  % U_aux = eigenvectors
vec_aux = diag(D_aux);      % vec_aux = eigenvalues

% Sort eigenvalues and eigenvectors (from highest to lowest)

[lin_vec,col_vec] = size(vec_aux);
[lin_val,col_val] = size(U_aux);
eig_val = zeros(lin_vec,col_vec);
eig_vec = zeros(lin_val,col_val);

for j = 1:M,
    eig_vec(:,j) = U_aux(:,M-j+1); 
    eig_val(j) = vec_aux(M-j+1);    
end

% Generate input matrix of samples mapped at feature space

Mphi = zeros(M,N);

for n = 1:N,
    xn = X(:,n);
    for i = 1:M,
        sum_aux = 0;
        for m = 1:M,
            zm = Mred(:,m);
            k_xz = kernel_func(xn,zm,PAR);
            sum_aux = sum_aux + eig_vec(m,i)*k_xz;
        end
        Mphi(i,n) = sum_aux / sqrt(eig_val(i));
    end
end

% Replaces Input matrix by mapped matrix

DATA.input = Mphi;
X = Mphi;

% After approximating, can use functions in data space.

PAR.Ktype = 0;

% Initialize Prototypes

Cx = prototypes_init(DATA,PAR);

% Verify if it is a decreasing neighboorhood function
if neig == 3,
    decay = 1;
else
    decay = 0;
end

% Main Loop

for ep = 1:Nep,

    % Save frame of the current epoch
    if (Von),
        VID(ep) = prototypes_frame(Cx,DATA);
    end
    
    % shuffle data
    I = randperm(N);
    X = X(:,I);
    
    % Update Neurons (one epoch)
    for i = 1:N,
        
        % Uptade Iteration
        t = t+1;

        % Update decreasing neighboorhood function of SOM
        [out_decay] = prototypes_decay(decay,Nn,neig,t,ep);
        Nn      = out_decay.Nn;
        neig    = out_decay.neig;
        t       = out_decay.t;
        
        % Get Winner Neuron in feature space (already mapped)
        xn = X(:,i);                                % Training Sample
        win = prototypes_win(Cx,xn,PAR);            % Winner Neuron Index
        n = prototypes_learn(learn,tmax,t,No,Nt);	% Learning Step
        
        % Uptade Neurons (Prototypes) in feature space (already mapped)
        for Nl = 1:Nk(1),
            for Nc = 1:Nk(2),
                % Current neuron (prototype)
                neu = [Nl Nc];
                % Calculate Neighborhood function
                h = som_f_neig(neig,win,neu,Nn,tmax,t,Vo,Vt);
                % Update function
                Cx(:,Nl,Nc) = Cx(:,Nl,Nc) + n*h*( xn - Cx(:,Nl,Nc) );
            end
        end

    end
    
    % SSE (one epoch)
    SSE(ep) = prototypes_sse(Cx,DATA,PAR);

end

% Assign indexes
for i = 1:N,
    xn = DATA.input(:,i);               % not shuffled data
    win = prototypes_win(Cx,xn,PAR);  	% Winner Neuron index
    ind(:,i) = win;                   	% save index for sample
end

% Hold Kernel type

PAR.Ktype = Ktype;

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.Cx = Cx;
PARout.ind = ind;
PARout.SSE = SSE;
PARout.VID = VID;
PARout.Mred = Mred;
PARout.Mphi = Mphi;
PARout.eig_vec = eig_vec;
PARout.eig_val = eig_val;

%% END