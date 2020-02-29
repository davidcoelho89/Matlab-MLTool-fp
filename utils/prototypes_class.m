function [OUT] = prototypes_class(DATA,PAR)

% --- Prototype-Based Classify Function ---
%
%   [OUT] = prototypes_class(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                  	[p x N]
%       PAR.
%           Cx = prototypes' attributes            	[p x Nk(1) x ... x Nk(Nd)]
%           Cy = prototypes' labels                 [Nc x Nk(1) x ... x Nk(Nd)]
%           K = number of nearest neighbors        	[cte]
%           dist = type of distance (if Ktype = 0) 	[cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance  
%               2: Euclidean distance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       OUT.
%           y_h = classifier's output                       [Nc x N]
%           win = closest prototype to each sample          [1 x N]

%% INITIALIZATION

% Data Initialization
X = DATA.input;                 % Input matrix
[~,N] = size(X);                % Number of samples

% Get Hyperparameters
K = PAR.K;                      % Number of nearest neighbors

% Prototypes and its labels
Cx = PAR.Cx;                 	% Prototype attributes
Cy = PAR.Cy;                 	% Prototype labels

% Vectorize prototypes and labels
Cx = prototypes_vect(Cx);
Cy = prototypes_vect(Cy);

% Problem Initilization
[Nc,Nk] = size(Cy);             % Number of prototypes and classes

% Init outputs
y_h = -1*ones(Nc,N);            % One output for each sample
win = zeros(1,N);               % One closest prototype for each sample

%% ALGORITHM

if (K == 1),        % if it is a nearest neighbor case
    
    for i = 1:N,
        % Display classification iteration (for debug)
        if(mod(i,1000) == 0)
            display(i);
        end
        
        % Get test sample
        sample = X(:,i);
        
        % Get closest prototype and min distance from sample to each class
        d_min = -1*ones(Nc,1);
        d_min_all = -1;
        for k = 1:Nk(1),
            prot = Cx(:,k);                         % Get prototype
            [~,class] = max(Cy(:,k));               % Get prototype label
            d = vectors_dist(prot,sample,PAR);      % Calculate distance
            if(d_min(class) == -1 || d < d_min(class)),
                d_min(class) = d;
            end
            if(d_min_all == -1 || d < d_min_all),   % Get closest prototype
                d_min_all = d;
                win(i) = k;
            end
        end
        
        % Fill output
        for class = 1:Nc,
            
            % Invert signal for second class in binary problems

            if(class == 2 && Nc == 2),
            	y_h(2,:) = -y_h(1,:);
                break;
            end
            
            % Calculate Class output for the sample
            
            dp = d_min(class);   % Get minimum distance from class
            if (dp == -1),      % no prototypes from class
                y_h(class,i) = -1;
            else
                dm = -1;        % get minimum distance from other classes
                for j = 1:Nc,
                    if(j == class),
                        continue;
                    elseif (dm == -1 || d_min(j) < dm),
                        dm = d_min(j);
                    end
                end
                if (dm == -1),  % no prototypes from other classes
                    y_h(class,i) = 1;
                else
                    y_h(class,i) = (dm - dp) / (dm + dp);
               end
            end
        end
        
    end
    
elseif (K > 1),    % if it is a knn case
    
    for i = 1:N,
        % Test sample
        sample = X(:,i);
        % Measure distance from sample to each prototype
        Vdist = zeros(1,Nk);
        for k = 1:Nk,
            prot = Cx(:,k);
            Vdist(k) = vectors_dist(prot,sample,PAR);
        end
        % sort distances and get nearest neighbors
        out = bubble_sort(Vdist,1);
        Knear = out.ind(1:K);
        % Find labels of nearest neighbors
        lbls = Cy(:,Knear);
        % Voting in order to find estimated label
        votes = zeros(1,Nc);
        for k = 1:K,
            [~,class] = max(lbls(:,k));
            votes(class) = votes(class) + 1;
        end
        % Update class
        [~,class] = max(votes);
        y_h(class,i) = 1;
    end
    
end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.win = win;

%% END