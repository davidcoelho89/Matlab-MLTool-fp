function [OUT] = prototypes_class(DATA,PAR)

% --- Prototype-Based Classify Function ---
%
%   [OUT] = prototypes_class(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%       PAR.
%           Cx = prototypes' attributes                         [p x Nk]
%           Cy = prototypes' labels                             [Nc x Nk]
%           K = number of nearest neighbors                     [cte]
%           knn_type = type of knn aproximation                 [cte]
%               1: Majority Voting
%               2: Weighted KNN
%           dist = type of distance (if Ktype = 0)              [cte]
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
%           y_h = classifier's output                           [Nc x N]
%           win = closest prototype to each sample              [1 x N]
%           dist = distance of sample from each prototype       [Nk x N]
%           near_ind = indexes for nearest prototypes           [K x N]

%% SET DEFAULT HYPERPARAMETERS

if (~(isfield(PAR,'K')))
    PAR.K = 1;
end
if (~(isfield(PAR,'knn_type')))
    PAR.knn_type = 1;
end

%% INITIALIZATIONS

% Data Initialization
X = DATA.input;                 % Input matrix
[~,N] = size(X);                % Number of samples

% Get Hyperparameters

K = PAR.K;                      % Number of nearest neighbors
knn_type = PAR.knn_type;        % Type of knn aproximation

% Prototypes and its labels
Cx = PAR.Cx;                 	% Prototype attributes
Cy = PAR.Cy;                 	% Prototype labels

% Problem Initilization
[Nc,Nk] = size(Cy);             % Number of prototypes and classes

% Init outputs
y_h = -1*ones(Nc,N);            % One output for each sample
winners = zeros(1,N);        	% One closest prototype for each sample
distances = zeros(Nk,N);        % Distance from prototypes to each sample

if (K == 1)
    nearest_indexes = zeros(1,N);
else
    if (Nk <= K)
        nearest_indexes = zeros(Nk,N);
    else
        nearest_indexes = zeros(K+1,N);
    end
end

%% ALGORITHM

if (K == 1)        % if it is a nearest neighbor case
    
    for n = 1:N
        
        % Display classification iteration (for debug)
        if(mod(n,1000) == 0)
            disp(n);
        end
        
        % Get test sample
        sample = X(:,n);
        
        % Get closest prototype and min distance from sample to each class
        d_min = -1*ones(Nc,1);
        d_min_all = -1;
        for k = 1:Nk(1)
            prot = Cx(:,k);                         % Get prototype
            [~,class] = max(Cy(:,k));               % Get prototype label
            d = vectors_dist(prot,sample,PAR);      % Calculate distance
            distances(k,n) = d;                     % hold distance
            if(d_min(class) == -1 || d < d_min(class))
                d_min(class) = d;
            end
            % Get closest prototype
            if(d_min_all == -1 || d < d_min_all)
                d_min_all = d;
                winners(n) = k;
                nearest_indexes(n) = k;
            end
        end
        
        % Fill output
        for class = 1:Nc
            
            % Invert signal for second class in binary problems

            if(class == 2 && Nc == 2)
            	y_h(2,:) = -y_h(1,:);
                break;
            end
            
            % Calculate Class output for the sample
            
            % Get minimum distance from class
            dp = d_min(class);
            % There is no prototypes from this class
            if (dp == -1)
                y_h(class,n) = -1;
            else
                % get minimum distance from other classes
                dm = -1;        
                for j = 1:Nc
                    if(j == class) % looking for other classes
                        continue;
                    elseif (d_min(j) == -1) % no prot from this class
                        continue;
                    elseif (dm == -1 || d_min(j) < dm)
                        dm = d_min(j);
                    end
                end
                if (dm == -1)  % no prototypes from other classes
                    y_h(class,n) = 1;
                else
                    y_h(class,n) = (dm - dp) / (dm + dp);
               end
            end
        end
        
    end
    
elseif (K > 1)    % if it is a knn case
    
    for n = 1:N
        
        % Display classification iteration (for debug)
        if(mod(n,1000) == 0)
            disp(n);
        end
        
        % Get test sample
        sample = X(:,n);
        
        % Measure distance from sample to each prototype
        Vdist = zeros(1,Nk);
        for k = 1:Nk
            prot = Cx(:,k);
            Vdist(k) = vectors_dist(prot,sample,PAR);
        end
        distances(:,n) = Vdist';    % hold distances
        
        % Sort distances and get nearest neighbors
        out = bubble_sort(Vdist,1);
        
        % Get closest prototype
        winners(n) = out.ind(1);
        
        % Verify number of prototypes and neighbors
        if(Nk <= K)
            nearest_indexes(:,n) = out.ind(1:Nk)';
            number_of_nearest = Nk;
        else
            nearest_indexes(:,n) = out.ind(1:K+1)';
            number_of_nearest = K;
        end
        
        % Get labels of nearest neighbors
        lbls_near = Cy(:,nearest_indexes(:,n)');
        
        if (knn_type == 1) % Majority voting method
            
            % Compute votes
            votes = zeros(1,Nc);
            for k = 1:number_of_nearest
                [~,class] = max(lbls_near(:,k));
                votes(class) = votes(class) + 1;
            end
            
            % Update class
            [~,class] = max(votes);
            y_h(class,n) = 1;

        else % Weighted knn
            
            % Avoid weights of 0
            epsilon = 0.001;
            
            % Auxiliary output and weight
            y_aux = zeros(Nc,1);
            w_sum = 0;
            
            % Get distances of nearest neighbors
            Dnear = Vdist(nearest_indexes(:,n)');
            
            % Calculate Output
            for k = 1:number_of_nearest
                % Compute Weight (Triangular)
                if (knn_type == 2)
                    Dnorm = Dnear(k)/(Dnear(end) + epsilon);
                    w = 1 - Dnorm;
                end
                w_sum = w_sum + w;
                % Compute weighted outptut
                y_aux = y_aux + w*lbls_near(:,k);

            end
            y_h(:,n) = y_aux / w_sum;
            
        end
        
    end
    
end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;
OUT.win = winners;
OUT.dist = distances;
OUT.near_ind = nearest_indexes;

%% END