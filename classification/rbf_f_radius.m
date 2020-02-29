function [r] = rbf_f_radius(C,PAR)

% --- Initialize radius / spread of each RBF ---
%
%   [r] = rbf_f_radius(C,PAR)
%
%   Input:
%       C = centroids                                   [p x Nh]
%       PAR.
%           Nh = number of hidden neurons (centroids)   [cte]
%           radius = type of radius / spread            [cte]
%               1: Equal for all RBFs
%               2: Each RBF has a different radius / spread
%   Output:
%       r = radius / spread of each rbf                 [1 x Nh]

%% INITIALIZATIONS

% Get Parameters

rad = PAR.rad;
q = PAR.Nh;

% Initialize radius with 0

r = zeros(1,q);

%% ALGORITHM

if (rad == 1)
    dmax = 0;
    for i = 1:q-1,            
        ci = C(:,i);                % get i-th centroid
        for j = i+1:q,      
            cj = C(:,j);            % get j-th centroid (i ~= j)
            d = sum((ci - cj).^2);  % Calculates distance
            if (d > dmax)
                dmax = d;           % gets maximum distance
            end
        end
    end
    r(1:q) = dmax / sqrt(2*q);      % all centroids are equal
    
elseif (rad == 2)
    for i = 1:q,
        ci = C(:,i);                % get i-th centroid
        dmin = 0;
        for j = 1:q,
            if (i == j),            % don´t get same centroid
                continue;
            end
            cj = C(:,j);            % get j-th centroid (i ~= j)
            d = sum((ci - cj).^2);  % Calculates distance
            if (dmin == 0 || d < dmin)
                dmin = d;           % gets minimum distance
            end
        end
        r(i) = dmin/2; % centroids are different from each one
    end

% ToDo - Implement heuristic No 3
    
else
    disp('Unknown initialization. Radius = 0.');
    
end

%% END