function [DATAout] = normalize(DATAin,OPTION)

% --- Normalize the raw data ---
%
%   [DATAout] = normalize(DATAin,OPTION)
%
%   Input:
%       DATAin.
%           input = data matrix                     [p x N]
%       OPTION.
%           norm = how will be the normalization
%               0: input_out = input_in
%               1: normalize between [0, 1]
%               2: normalize between [-1, +1]
%               3: normalize by z-score transformation
%                  (empirical mean = 0 and standard deviation = 1)
%                  Xnorm = (X-Xmean)/(std)
%   Output:
%       DATAout.
%           input = normalized matrix               [p x N]

%% INITIALIZATIONS

option = OPTION.norm;   % gets normalization option from structure
X = DATAin.input';      % gets and transpose data from structure - [N x p]

[N,p] = size(X);        % number of samples and attributes
Xmin = min(X)';         % minimum value of each attribute
Xmax = max(X)';         % maximum value of each attribute
Xmed = mean(X)';        % mean of each attribute
dp = std(X)';           % standard deviation of each attribute

%% ALGORITHM

X_norm = zeros(N,p); % initialize data

switch option
    case (0)
        X_norm = X;
    case (1)    % normalize between [0 e 1]
        for i = 1:N,
            for j = 1:p,
                X_norm(i,j) = (X(i,j) - Xmin(j))/(Xmax(j) - Xmin(j)); 
            end
        end
    case (2)    % normalize between [-1 e +1]
        for i = 1:N,
            for j = 1:p,
                X_norm(i,j) = 2*(X(i,j) - Xmin(j))/(Xmax(j) - Xmin(j)) - 1; 
            end
        end
    case (3)    % normalize by z-score transform (by mean and std)
        for i = 1:N,
            for j = 1:p,
                X_norm(i,j) = (X(i,j) - Xmed(j))/dp(j); 
            end
        end
    otherwise
        X_norm = X;
        disp('Choose a correct option. Data was not normalized.')
end

X_norm = X_norm'; % transpose data for [p x N] pattern

%% FILL OUTPUT STRUCTURE

DATAin.input = X_norm;
DATAin.Xmin = Xmin;
DATAin.Xmax = Xmax;
DATAin.Xmed = Xmed;
DATAin.dp = dp;

DATAout = DATAin;

%% END