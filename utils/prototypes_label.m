function [PARout] = prototypes_label(DATA,OUT_CL)

% --- Clusters' Labeling Function ---
%
%   [PARout] = prototypes_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x Ntr]
%           output = output matrix                              [Nc x Ntr]
%       OUT_CL.
%           Cx = cluster prototypes                             [p x Nk]
%           dist = type of distance                             [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           ind = cluster index for each sample                 [1 x Ntr]
%           lbl = type of labeling                              [cte]
%               1: Majority voting
%               2: Average distance
%               3: Minimum distance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PARout.
%           Cx = clusters prototypes                            [p x Nk]
%           Cy = class of each prototype                        [Nc x Nk]

%% INITIALIZATION

% Get data and classes
X = DATA.input;
Y = DATA.output;

% Get number of samples and classes
[Nc,N] = size(Y);

% Convert Labels for sequential pattern [1, 2, ... , Nc]
[~,Y] = max(Y);

% Get prototypes and labeling type
Cx = OUT_CL.Cx;
lbl_type = OUT_CL.lbl;

% Get number of prototypes
[~,Nk] = size(Cx);         	

% Init other parameters
counter = zeros(Nk,Nc);   	% for average distance and voronoi labeling
mean_dist = zeros(Nk,Nc);	% for average distance labeling
min_dist = 0;            	% for minimum distance labeling

%% ALGORITHM 

% init output
label = zeros(1,Nk);

if lbl_type == 1,       % Majority Voting Method
    
    % Fill voting matrix
    for n = 1:N,
        xn = X(:,n);                                % data sample
        win = prototypes_win(Cx,xn,OUT_CL);         % winner neuron index
        counter(win,Y(n)) = counter(win,Y(n)) + 1;	% add to voting matrix
    end
    
    % Set Labels
    for k = 1:Nk(1),
        [~,class] = max(counter(k,:));	% Get class with max no of votes
        label(k) = class;               % label prototypes
    end
    
elseif lbl_type == 2,	% Average Distance Method
    
	for n = 1:N,
        % Get sample
        xn = X(:,n);
        % Find winner prototype
        win = prototypes_win(Cx,xn,OUT_CL);
        cx = Cx(:,win);
        % Calculate distance
        dist_curr = vectors_dist(cx,xn,OUT_CL);
        % Update mean and counter
        counter(win,Y(n)) = counter(win,Y(n)) + 1;
        mean_dist(win,Y(n)) = mean_dist(win,Y(n)) + dist_curr;
	end
    
    % Verify class with minimum mean distance
    for k = 1:Nk,
        min_dist = 0;   % init minimum distance with 0
        label(k) = 1;   % init class with 1
        for c = 1:Nc,
            if counter(k,c) ~= 0,
                % Calculate mean distance
                mean_dist(k,c) = mean_dist(k,c) / counter(k,c);
                % Update minimum mean distance
                if(min_dist == 0 || mean_dist(k,c) < min_dist),
                    min_dist = mean_dist(k,c);
                    label(k) = c;
                end 
            end
        end
    end
    
elseif lbl_type == 3,   % Minimum Distance Method
    
    for k = 1:Nk,
        for n = 1:N,
            % Get Sample, neuron and current distance
            xn = X(:,n);
            cx = Cx(:,k);                        
            dist_curr = vectors_dist(cx,xn,OUT_CL);
            % set label and minimal distance
            if n == 1,
                min_dist = dist_curr;
                label(k) = Y(n);
            else
                if dist_curr < min_dist,
                    min_dist = dist_curr;
                    label(k) = Y(n);
                end
            end
        end
    end
    
end

% Convert to [-1 +1 -1] pattern
label_aux = -1*ones(Nc,Nk);
for k = 1:Nk,
    label_aux(label(k),k) = 1;
end

%% FILL OUTPUT STRUCTURE

PARout = OUT_CL;            % Get parameters from training data
PARout.Cy = label_aux;      % Set neuron's labels

%% END