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
%           K = number of nearest neighbors (classify)        	[cte]
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
%       	Cx = clusters' centroids (prototypes)               [p x Nk]
%           Cy = clusters' labels                               [Nc x Nk]
%           Km = Kernel Matrix of Entire Dictionary             [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel Matrix of Dictionary          [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]
%           VID = frame struct (played by 'video function')     [1 x Nep]
%           y_h = class prediction                              [Nc x N]
 
 PAR = DATA + HP;