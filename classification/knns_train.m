function [PAR] = knns_train(DATA,HP)
 
% --- Sliding Window KNN Training Function ---
%
%   [PAR] = knns_train(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       HP.
%           dist = type of distance                             [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance%   Output:
%           Ws = window size                                    [cte]
%           K = number of nearest neighbors (classify)        	[cte]
%           knn_type = type of knn aproximation                 [cte]
%               1: Majority Voting
%               2: Weighted KNN
%   Output:
%       PAR.
%       	Cx = clusters' centroids (prototypes)               [p x Nk]
%           Cy = clusters' labels                               [Nc x Nk]
%           y_h = class prediction                              [Nc x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP)))
    PARaux.dist = 2;        % Type of distance = euclidean
    PARaux.Ws = 100;        % Window size of 100
    PARaux.Ktype = 0;       % Non-kernelized Algorithm
    PARaux.K = 5;           % Number of nearest neighbors = 5
    PARaux.knn_type = 1;    % Majority voting KNN
    HP = PARaux;
else
    if (~(isfield(HP,'dist')))
        HP.dist = 2;
    end
    if (~(isfield(HP,'Ws')))
        HP.Ws = 100;
    end
    if (~(isfield(HP,'Ktype')))
        HP.Ktype = 0;
    end
    if (~(isfield(HP,'K')))
        HP.K = 5;
    end
    if (~(isfield(HP,'knn_type')))
        HP.knn_type = 1;
    end
end

%% INITIALIZATIONS

% Data Initialization

X = DATA.input;         % Input Matrix
Y = DATA.output;        % Output Matrix

% Get Hyperparameters

Ws = HP.Ws;             % Windows Size

% Problem Initialization

[Nc,N] = size(Y);       % Total of classes and samples

% Init Outputs

PAR = HP;

if (~isfield(PAR,'Cx'))
    PAR.Cx = [];
    PAR.Cy = [];
end

yh = -1*ones(Nc,N);

%% ALGORITHM

for n = 1:N
    
    % Get sample
    DATAn.input = X(:,n);
	DATAn.output = Y(:,n);
    
    % Get dictionary size (cardinality, number of prototypes)
	[~,Nk] = size(PAR.Cx);
    
    % Init Dictionary (if it is the first sample)
    if(Nk == 0)
        % Make a guess (yh = 1 => first class)
        yh(1,n) = 1;
        % Add sample to dictionary
        PAR.Cx = DATAn.input;
        PAR.Cy = DATAn.output;
        % Calls next sample
        continue;
    end
    
    % Predict output
    OUTn = knns_classify(DATAn,PAR);
    yh(:,n) = OUTn.y_h;
    
    % Update window
    if (Nk < Ws)
        PAR.Cx = [DATAn.input, PAR.Cx];
        PAR.Cy = [DATAn.output, PAR.Cy];
    else
        PAR.Cx = [DATAn.input, PAR.Cx(:,1:end-1)];
        PAR.Cy = [DATAn.output, PAR.Cy(:,1:end-1)];
    end
    
end

%% FILL OUTPUT STRUCTURE

PAR.y_h = yh;

%% END