function [DB] = index_db(DATA,PAR)

% ---  Calculate Davies-Bouldin index for Clustering ---
%
%   [DB] = index_db(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                     	[p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           ind = cluster index for each sample         [1 x N]
%           SSE = Sum of Squared Errors for each epoch 	[1 x Nep]
%   Output:
%       DB = DB index                                   [cte]

%% INIT

% Load Data

X = DATA.input;

% Load Parameters

indexes = PAR.ind;
C = PAR.Cx;
[~,Nk] = size(C);

% Init Aux Variables

clusters = cell(1,Nk);
for k = 1:Nk,
    clusters{k} = X(:,indexes == k);
end

%% ALGORITHM

if Nk == 1,
    DB = NaN;
else
    DB = 0;                 % DB accumulator
    sigma = zeros(1,Nk);    % find standard deviation of each cluster
    
    for k = 1:Nk
        % Find distance of objects at each cluster to its centroid
        ck = C(:,k);            % Get Centroid
        Xk = clusters{k};       % Get samples
        [~,Nsamples] = size(Xk);
        dispersion = zeros(1,Nsamples);
        for n = 1:Nsamples
            xn = Xk(:,n);
            dispersion(n) = sum((ck - xn).^2);
        end
        % Find sigma
        sigma(k) = sqrt(sum(dispersion)/Nsamples);
    end
    
    % Find max((sigma_k+sigma_m/d_km))
    for k = 1:Nk
        aux_value = zeros(1,Nk);
        ck = C(:,k);
        for m = 1:Nk
            if m ~= k
                cm = C(:,m);
                d_km = sqrt(sum((ck - cm).^2));
                aux_value(m) = (sigma(k) + sigma(m))/d_km;
            end
        end
        % Find DB index
        DB = DB + (max(aux_value))/Nk;
    end
end

%% FILL OUTPUT STRUCTURE

% Dont Need. Output is just a constant.

%% END