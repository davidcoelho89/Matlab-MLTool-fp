 function [PAR] = k2nn_train(DATA,HP)

% --- Kernel KNN Prototype-Based Training Function ---
%
%   PAR = k2nn_train(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       HP.
%           Dm = Design Method                                  [cte]
%               = 1 -> all data set
%               = 2 -> per class
%           Ss = Sparsification strategy                        [cte]
%               = 1 -> ALD
%               = 2 -> Coherence
%               = 3 -> Novelty
%               = 4 -> Surprise
%           v1 = Sparseness parameter 1                         [cte]
%           v2 = Sparseness parameter 2                         [cte]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method 1
%               = 2 -> score-based method 2
%               = 2 -> Matching-pursuit (rem prot, verify error)
%               = 3 -> Penalization (build lssvm, rem prot)
%               = 4 -> FBS - forward-backward
%           min_score = score that leads to prune prototype     [cte]
%           Us = Update strategy                                [cte]
%               = 0 -> do not update prototypes
%               = 1 -> lms (wta)
%               = 2 -> lvq (supervised)
%           eta = Update rate                                   [cte]
%           max_prot = max number of prototypes ("Budget")      [cte]
%           Von = enable or disable video                       [cte]
%           K = number of nearest neighbors (classify)        	[cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PAR.
%       	Cx = clusters' centroids (prototypes)               [p x Nk]
%           Cy = clusters' labels                               [Nc x Nk]
%           Km = Kernel Matrix of Dictionary                    [Nk x Nk]
%           Kinv = Inverse Kernel Matrix of Dictionary          [Nk x Nk]
%           score = used for prunning method                    [1 x Nk]
%           class_hist = used for prunning method               [1 x Nk]
%           ind = cluster index for each sample                 [1 x N]
%           VID = frame struct (played by 'video function')     [1 x Nep]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    PARaux.Dm = 2;          % Design Method
    PARaux.Ss = 1;          % Sparsification strategy
    PARaux.v1 = 0.1;        % Sparseness parameter 1 
    PARaux.v2 = 0.9;        % Sparseness parameter 2
    PARaux.Ps = 0;          % Prunning strategy
    PARaux.min_score = -10; % Score that leads to prune prototype
    PARaux.Us = 0;          % Update strategy
    PARaux.eta = 0.01;      % Update Rate
    PARaux.max_prot = Inf;  % Max number of prototypes
    PARaux.Von = 0;         % enable / disable video
    PARaux.K = 1;           % Number of nearest neighbors (classify)
    PARaux.Ktype = 2;       % Kernel Type (gaussian)
    PARaux.sig2n = 0.001;   % Kernel regularization parameter
    PARaux.sigma = 2;       % Kernel width (gaussian)
    PARaux.gamma = 2;       % Polynomial order
    PARaux.alpha = 1;       % Dot product multiplier
    PARaux.theta = 1;       % Dot product add cte 
	HP = PARaux;
else
    if (~(isfield(HP,'Dm'))),
        HP.Dm = 2;
    end
    if (~(isfield(HP,'Ss'))),
        HP.Ss = 1;
    end
    if (~(isfield(HP,'v1'))),
        HP.v1 = 0.1;
    end
    if (~(isfield(HP,'v2'))),
        HP.v2 = 0.9;
    end
    if (~(isfield(HP,'Ps'))),
        HP.Ps = 0;
    end
    if (~(isfield(HP,'min_score'))),
        HP.min_score = -10;
    end
    if (~(isfield(HP,'Us'))),
        HP.Us = 0;
    end
    if (~(isfield(HP,'eta'))),
        HP.eta = 0.01;
    end
    if (~(isfield(HP,'max_prot'))),
        HP.max_prot = Inf;
    end
    if (~(isfield(HP,'Von'))),
        HP.Von = 0;
    end
    if (~(isfield(HP,'K'))),
        HP.K = 1;
    end
    if (~(isfield(HP,'Ktype'))),
        HP.Ktype = 2;
    end
    if (~(isfield(HP,'sig2n'))),
        HP.sig2n = 0.001;
    end
    if (~(isfield(HP,'sigma'))),
        HP.sigma = 2;
    end
    if (~(isfield(HP,'gamma'))),
        HP.gamma = 2;
    end
    if (~(isfield(HP,'alpha'))),
        HP.alpha = 1;
    end
    if (~(isfield(HP,'theta'))),
        HP.theta = 1;
    end
end

%% INITIALIZATIONS

% Data Initialization

X = DATA.input;         % Input Matrix
Y = DATA.output;        % Output Matrix

% Get Hyperparameters

Von = HP.Von;
max_prot = HP.max_prot;

% Problem Initialization

[~,N] = size(X);       	% Total of samples

% Init Outputs

if (isfield(HP,'Cx'))
    D.x = HP.Cx;
    D.y = HP.Cy;
else
    D.x = [];
    D.y = [];
end

if (isfield(HP,'Km'))
    D.Km = HP.Km;
    D.Kinv = HP.Kinv;
else
    D.Km = [];
    D.Kinv = [];
end

if (isfield(HP,'score'))
    D.score = HP.score;
    D.class_hist = HP.class_hist;
else
    D.score = [];
    D.class_hist = [];
end

ind = zeros(1,N);

VID = struct('cdata',cell(1,N),'colormap', cell(1,N));

%% ALGORITHM

for t = 1:N,
    
    % Display samples (for debug)
%     if(mod(t,10000) == 0)
%         display(t);
%     end

    % Save frame of the current epoch
    if (Von),
        VID(t) = prototypes_frame(D.x,DATA);
    end
    
    % Get sample
    xt = X(:,t);
    yt = Y(:,t);

    % Get the size of the dictionary
    [~,mt1] = size(D.x);
    
    % Dont Add if number of prototypes is too high
    if (mt1 > max_prot),
        break;
    end

    % Apply sparsification strategy
    [D] = k2nn_dict_grow(xt,yt,D,HP);
    
    % Get the new size of the dictionary
    [~,mt2] = size(D.x);
    
    % Apply prunning strategy
    [D] = k2nn_dict_prun(D,HP);

    % Verify number of prototypes
    if ((mt2 - mt1) ~= 1)
     	% Apply update strategy
        [D] = k2nn_dict_updt(xt,yt,D,HP);
    else
     	display(mt2); % Debug
     end
    
end

% Assign indexes (the algorithm becomes slow with the following lines)
% for i = 1:N,
%     xn = DATA.input(:,i);               % not shuffled data
%     win = prototypes_win(D.x,xn,PAR);  	% Winner Neuron index
%     ind(:,i) = win;                   	% save index for sample
% end

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = D.x;
PAR.Cy = D.y;
PAR.Km = D.Km;
PAR.Kinv = D.Kinv;
PAR.score = D.score;
PAR.class_hist = D.class_hist;
PAR.ind = ind;
PAR.VID = VID;

%% END