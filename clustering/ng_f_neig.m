function [h] = ng_f_neig(C,sample,PAR,t,tmax)

% --- NG Neighborhood Function ---
%
%   [h] = ng_f_neig(C,sample,dist,neig,tmax,t,Lo,Lt)
%
%   Input:
%       C = clusters centroids (prototypes)             [p x Nk]
%       sample = current pattern                        [p x 1]
%       PAR.
%           dist = type of distance                    	[cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           neig = type of neighborhood function      	[cte]
%               1: L = Lo (constant)
%               2: L = Lo*((Lt/Lo)^(t/tmax))
%           Lo = initial value of lambda               	[cte]
%           Lt = final value of lambda               	[cte]
%       t = current iteration                           [cte]
%       tmax = max number of iterations                 [cte]
%   Output:
%       h = neigborhood function result                 [cte]

%% INITIALIZATIONS

[~,Nk] = size(C);   % number of prototypes
dist = PAR.dist;    % type of distance 
neig = PAR.neig;    % type of neighborhood function
Lo = PAR.Lo;        % initial value of lambda
Lt = PAR.Lt;        % final value of lambda

%% ALGORITHM

% Calculate distances from sample

Vdist = zeros(1,Nk);

for prot = 1:Nk
    
    % Get prototype
    prototype = C(:,prot);
    
    % Get Distance
    Vdist(prot) = vectors_dist(sample,prototype,PAR);

end

% Calculate ranks

rank = zeros(1,Nk);
for prot = 1:Nk
    for j = 1:Nk
        if (Vdist(prot) > Vdist(j))
            rank(prot) = rank(prot) + 1;
        end
    end
end

% Reverse rank if dot product

if (dist == 0)
    rank_aux = zeros(1,Nk);
    for i = 1:Nk
        rank_aux(i) = rank(Nk - i + 1);
    end
    rank = rank_aux;
end

% Calculate neighborhood function

h = zeros(1,Nk);
if neig == 1
    for i = 1:Nk
        L = Lo;
        h(i) = exp(-rank(i)/L);
    end
elseif neig == 2
    for i = 1:Nk
        L = Lo*((Lt/Lo)^(t/tmax));
        h(i) = exp(-rank(i)/L);
    end
end

%% END