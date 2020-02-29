function [S] = index_silhouette(DATA,PAR)

% ---  Calculate silhuette index for Clustering ---
%
%   [S] = index_silhouette(C,PAR,N)
%
%   Input:
%       DATA.
%           input = input matrix                     	[p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           ind = cluster index for each sample         [1 x N]
%           SSE = Sum of Squared Errors for each epoch 	[1 x Nep]
%   Output:
%       S = Silhouette index                            [cte]

%% INIT

% Load Data

X = DATA.input;
[~,N] = size(X);

% Load Parameters

indexes = PAR.ind;
Nk = length(find(unique(indexes)));

% Init Aux Variables

clusters = cell(1,Nk);
cohesion = cell(1,Nk);
separation = cell(1,Nk);
Nsamples = zeros(1,Nk);

for k = 1:Nk,
    clusters{k} = X(:,indexes == k);
    Nsamples(k) = length(find(indexes == k)); 
    cohesion{k} = zeros(1,Nsamples(k));
    separation{k} = zeros(1,Nsamples(k));
end

%% ALGORITHM

if Nk == 1,
    S = NaN;
else
    S = 0; % Silhouette accumulator
    
    for k = 1:Nk,
        Xk = clusters{k};
        for m = 1:Nsamples(k)
            x1 = Xk(:,m);
            %Find Cohesion
            for n = 1:Nsamples(k)
                if m ~= n
                    x2 = Xk(:,n);
                    cohesion{k}(m) = cohesion{k}(m) + ...
                                     (sqrt(sum((x1-x2).^2))/(Nsamples(k) - 1));
                end
            end
            
            %Find Separation
            separation_aux = zeros(1,Nk);
            for o = 1:Nk
                if o == k
                    separation_aux(o) = inf;
                else
                    for n = 1:Nsamples(o)
                        x2 = clusters{o}(:,n);
                        separation_aux(o) = separation_aux(o) + ...
                                            (sqrt(sum((x1-x2).^2))/Nsamples(o));
                    end
                end
            end
            separation{k}(m) = min(separation_aux);
            
            %Find Silhouette Coefficient
            dif_value = separation{k}(m) - cohesion{k}(m);
            max_value = max(separation{k}(m),cohesion{k}(m));
            s = (dif_value/max_value)/N;
            S = S + s;
            
        end
    end
end

%% FILL OUTPUT STRUCTURE

% Dont Need. Output is just a constant.

%% END