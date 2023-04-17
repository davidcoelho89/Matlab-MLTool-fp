function [STATS] = cluster_stats_1turn(DATA,OUT_CL)

% --- Provide Statistics of 1 turn of Clustering ---
%
%   [STATS] = cluster_stats_1turn(DATA,OUT_CL)
% 
%   Input:
%    	DATA.
%           input = input vectors                       [p x N]
%       OUT_CL.
%     		Cx = clusters centroids (prototypes)        [p x Nk]
%           ind = cluster index for each sample         [1 x N]
%           SSE = Sum of Squared Errors for each epoch	[1 x Nep]           
%   Output:
%       STATS.
%           ssqe = sum of squared quantization errors	[cte]
%           msqe = mean squared quantization errors     [cte]
%           aic = Akaike Info Criterion  index          [cte]
%           bic = Bayesian Info Criterion index     	[cte]
%           ch = ch index                               [cte]
%           db = Davies-Bouldin index                   [cte]
%           dunn = dunn index                           [cte]
%           fpe = fpe index                             [cte]
%           mdl = Minimum Description Length index   	[cte]
%           sil = Silhouette index                      [cte]

%% INITIALIZATIONS

% Get data and prototypes

X = DATA.input;         % input vectors
ind = OUT_CL.ind;       % cluster index for each sample
Cx = OUT_CL.Cx;        	% prototypes

[p,N] = size(X);        % number of samples
dim = size(Cx);       	% dimensions of prototypes grid
c_dim = dim(2:end);     % don't use attributes' dimension

% Init metrics

SSQE = 0;               % Sum of squared quantization errors
MSQE = 0;               % Mean squared quantization error

%% ALGORITHM

% SSQE and MSQE

for i = 1:N
    xn = X(:,i);                        	% get sample
    if(length(c_dim) == 1)
        cx = Cx(:,ind(1,i));              	% get prototype
    elseif (length(c_dim) == 2)
        cx = zeros(p,1);
        cx(1:p) = Cx(:,ind(1,i),ind(2,i));	% get prototype
    end
    SSQE = SSQE + sum((xn - cx).^2);
end

MSQE = MSQE + SSQE / N;

% CLUSTERING METRICS

AIC = index_aic(DATA,OUT_CL);
BIC = index_bic(DATA,OUT_CL);
CH = index_ch(DATA,OUT_CL);
DB = index_db(DATA,OUT_CL);
DUNN = index_dunn(DATA,OUT_CL);
FPE = index_fpe(DATA,OUT_CL);
MDL = index_mdl(DATA,OUT_CL);
SIL = index_silhouette(DATA,OUT_CL);

% Medidas intermediarias: Sb, Sw

%% FILL OUTPUT STRUCTURE

STATS.ssqe = SSQE;
STATS.msqe = MSQE;
STATS.aic = AIC;
STATS.bic = BIC;
STATS.ch = CH;
STATS.db = DB;
STATS.dunn = DUNN;
STATS.fpe = FPE;
STATS.mdl = MDL;
STATS.sil = SIL;

%% END