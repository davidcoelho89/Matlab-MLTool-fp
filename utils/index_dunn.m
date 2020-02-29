function [DUNN] = index_dunn(DATA,PAR)

% ---  Calculate Dunn index for Clustering ---
%
%   [DUNN] = index_dunn(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                     	[p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           ind = cluster index for each sample         [1 x N]
%           SSE = Sum of Squared Errors for each epoch 	[1 x Nep]
%   Output:
%       DUNN = DUNN index                               [cte]

%% INIT

% Load Data

X = DATA.input;

% Load Parameters

indexes = PAR.ind;
C = PAR.Cx;
[~,Nk] = size(C);

% Init Aux Variables

clusters = cell(1,Nk);
Nsamples = zeros(1,Nk);

for k = 1:Nk,
    clusters{k} = X(:,indexes == k);
    Nsamples(k) = length(find(indexes == k));
end

%% ALGORITHM

if Nk == 1,
    DUNN = NaN;
else
    %Find centroids combination
    C_comb = combvec(1:Nk,1:Nk);
    comb_aux = C_comb(1,:) - C_comb(2,:);
    C_comb = C_comb(:,comb_aux ~= 0);
    [~,Ncomb] = size(C_comb);
    
    %Find min(Sigma(Si,Sj))
    min_sigma = inf;
    for i = 1:Ncomb
        for j = 1:Nsamples(C_comb(1,i))
            Si = clusters{C_comb(1,i)}(:,j);
            for n = 1:Nsamples(C_comb(2,i))
                Sj = clusters{C_comb(2,i)}(:,n);
                deltaSiSj = sqrt(sum((Si-Sj).^2));
                if deltaSiSj < min_sigma
                    min_sigma = deltaSiSj;
                end
            end
        end
    end
    
    den = 0;
    for k = 1:Nk
        Xk = clusters{k};
        for n = 1:Nsamples(k)
            x = Xk(:,n);
            for m = 1:Nsamples(k)
                if m ~= n
                    y = Xk(:,m);
                    Sl = sqrt(sum((x - y).^2));
                    if Sl > den
                        den = Sl;
                    end
                end
            end
        end
    end
    
    DUNN = min_sigma/den;
end

%% FILL OUTPUT STRUCTURE

% Dont Need

%% END