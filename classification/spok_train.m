 function [PAR] = spok_train(DATA,HP)
 
% --- SParse Online adptive Kernel Training Function ---
%
%   [PAR] = spok_train(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       HP.
%           Ne = maximum number of epochs	                    [cte]
%           is_istatic = Verify if the dataset is stationary    [0 or 1]
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
%           update_kernel_matrix = 0 or 1                       [cte]
%               = depends on Ss. Ex: "=1" for ALD and Surprise.
%           Us = Update strategy                                [cte]
%               = 0 -> do not update prototypes
%               = 1 -> wta (lms, unsupervised)
%               = 2 -> lvq (supervised)
%           eta = Update rate                                   [cte]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method 1 (drift based)
%               = 2 -> score-based method 2 (hits and errors)
%           min_score = score that leads to prune prototype     [cte]
%           max_prot = max number of prototypes ("Budget")      [cte]
%           min_prot = min number of prototypes ("restriction") [cte]
%           Von = enable or disable video                       [cte]
%           K = number of nearest neighbors (classification)   	[cte]
%           knn_type = type of knn aproximation                 [cte]
%               1: Majority Voting
%               2: Weighted KNN
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PAR.
%       	Cx = clusters' centroids (prototypes)               [p x Q]
%           Cy = clusters' labels                               [Nc x Q]
%           Km = Kernel Matrix of Entire Dictionary             [Q x Q]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel Matrix of Dictionary          [Q x Q]
%           Kinvc = Inv Kernel Matrix for each class (cell)     [Nc x 1]
%           score = used for prunning method                    [1 x Q]
%           class_history = used for prunning method           	[1 x Q]
%           times_selected = used for prunning method           [1 x Q]
%           times_selected_sum = used for debug                 [cte]
%           VID = frame struct (played by 'video function')     [1 x Nep]
%           y_h = class prediction                              [Nc x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP)))
    PARaux.Ne = 1;          % Maximum number of epochs
    PARaux.is_static = 1;   % Verify if the data set is stationary
    PARaux.Dm = 2;          % Design Method
    PARaux.Ss = 1;          % Sparsification strategy
    PARaux.v1 = 0.1;        % Sparseness parameter 1 
    PARaux.v2 = 0.9;        % Sparseness parameter 2
    PARaux.Us = 1;          % Update strategy
    PARaux.eta = 0.1;       % Update Rate
    PARaux.Ps = 2;          % Prunning strategy
    PARaux.min_score = -10; % Score that leads to prune prototype
    PARaux.max_prot = Inf;  % Max number of prototypes
    PARaux.min_prot = 1;    % Min number of prototypes
    PARaux.Von = 0;         % Enable / disable video
    PARaux.K = 1;           % Number of nearest neighbors (classify)
    PARaux.knn_type = 2;    % Distance Weighted knn
    PARaux.Ktype = 2;       % Kernel Type (gaussian)
    PARaux.sig2n = 0.001;   % Kernel regularization parameter
    PARaux.sigma = 2;       % Kernel width (gaussian)
    PARaux.alpha = 1;       % Dot product multiplier
    PARaux.theta = 1;       % Dot product add cte 
    PARaux.gamma = 2;       % Polynomial order
	HP = PARaux;
else
    if (~(isfield(HP,'Ne')))
        HP.Ne = 1;
    end
    if (~(isfield(HP,'is_static')))
        HP.is_static = 1;
    end
    if (~(isfield(HP,'Dm')))
        HP.Dm = 2;
    end
    if (~(isfield(HP,'Ss')))
        HP.Ss = 1;
    end
    if (~(isfield(HP,'v1')))
        HP.v1 = 0.1;
    end
    if (~(isfield(HP,'v2')))
        HP.v2 = 0.9;
    end
    if (~(isfield(HP,'Us')))
        HP.Us = 1;
    end
    if (~(isfield(HP,'eta')))
        HP.eta = 0.1;
    end
    if (~(isfield(HP,'Ps')))
        HP.Ps = 2;
    end
    if (~(isfield(HP,'min_score')))
        HP.min_score = -10;
    end
    if (~(isfield(HP,'max_prot')))
        HP.max_prot = Inf;
    end
    if (~(isfield(HP,'min_prot')))
        HP.min_prot = 1;
    end
    if (~(isfield(HP,'Von')))
        HP.Von = 0;
    end
    if (~(isfield(HP,'K')))
        HP.K = 1;
    end
    if (~(isfield(HP,'knn_type')))
        HP.knn_type = 2;
    end
    if (~(isfield(HP,'Ktype')))
        HP.Ktype = 2;
    end
    if (~(isfield(HP,'sig2n')))
        HP.sig2n = 0.001;
    end
    if (~(isfield(HP,'sigma')))
        HP.sigma = 2;
    end
    if (~(isfield(HP,'alpha')))
        HP.alpha = 1;
    end
    if (~(isfield(HP,'theta')))
        HP.theta = 1;
    end
    if (~(isfield(HP,'gamma')))
        HP.gamma = 2;
    end
end

%% INITIALIZATIONS

% Data Initialization

X = DATA.input;             % Input Matrix
Y = DATA.output;            % Output Matrix

% Get Hyperparameters

Ne = HP.Ne;                 % Maximum number of epochs
Von = HP.Von;               % Enable or not Video
is_static = HP.is_static;   % Verify if data set is static

% Problem Initialization

[Nc,N] = size(Y);           % Total of classes and samples

% Init Outputs

PAR = HP;

if (~isfield(PAR,'Cx'))

    PAR.Cx = [];
    PAR.Cy = [];
    
    PAR.Km = [];
    PAR.Kmc = [];
    PAR.Kinv = [];
    PAR.Kinvc = [];
    
    PAR.score = [];
    PAR.class_history = [];
    PAR.times_selected = [];
    PAR.times_selected_sum = 0;
    
    if(PAR.Ss == 1 || PAR.Ss == 4)
        PAR.update_kernel_matrix = 1;
    else
        PAR.update_kernel_matrix = 0;
    end
    
end

VID = struct('cdata',cell(1,N*Ne),'colormap', cell(1,N*Ne));
it = 0;

yh = -1*ones(Nc,N);

%% ALGORITHM

% Update Dictionary

for epoch = 1:Ne
    
    for n = 1:N

        % Save frame of the current iteration
        if (Von)
            it = it+1;
            VID(it) = prototypes_frame(PAR.Cx,DATA);
        end

        % Get sample
        DATAn.input = X(:,n);
        DATAn.output = Y(:,n);

        % Get dictionary size (cardinality, number of prototypes)
        [~,Qt1] = size(PAR.Cx);

        % Init Dictionary (if it is the first sample)
        if (Qt1 == 0)
            % Make a guess (yh = [1 -1 -1 ... -1 -1]' : first class)
            yh(1,n) = 1;
            % Add sample to dictionary
            PAR = spok_dict_grow(DATAn,PAR);
            % Calls next sample
            continue;
        end

        % Predict Output
        OUTn = spok_classify(DATAn,PAR);
        yh(:,n) = OUTn.y_h;

        % Update number of times a prototype has been selected
        % (as the winner)
        win = OUTn.win;
        PAR.times_selected(win) = PAR.times_selected(win) + 1;

        % Growing Strategy
        PAR = spok_dict_grow(DATAn,PAR);

        % Get dictionary size (cardinality, number of prototypes)
        [~,Qt2] = size(PAR.Cx);

        % Update Strategy (if prototype was not added)
        if(Qt2 - Qt1 == 0)
            PAR = spok_dict_updt(DATAn,PAR);
        else
            % For debug. Display dictionary size when it grows.
            % display(Qt2);
        end

        % Prunning Strategy (heuristic based)
        PAR = spok_score_updt(DATAn,OUTn,PAR);
        PAR = spok_dict_prun(PAR);

    end
    
    if(is_static)
        
        % Shuffle Data
        I = randperm(N);        
        X = X(:,I);     
        Y = Y(:,I);
        
        % Hold last classification labels
        if (epoch == Ne)
            OUT = spok_classify(DATA,PAR);
            yh = OUT.y_h;
        end
        
    end

end

%% FILL OUTPUT STRUCTURE
 
PAR.VID = VID;
PAR.y_h = yh;

end