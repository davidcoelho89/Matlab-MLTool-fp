function [DATAout] = denormalize(DATAin,OPTION)

% --- Denormalize the normalized data ---
%
%   [DATAout] = denormalize(DATAin,OPTION)
%
%   Input:
%       DATAin.
%           input = data matrix                     [p x N]
%       OPTION.
%           norm = how will be the normalization
%               0: input_out = input_in
%               1: denormalize between [0, 1]
%               2: denormalize between [-1, +1]
%               3: denormalize by by z-score transformation
%                  (empirical mean = 0 and standard deviation = 1)
%                  Xnorm = (X-Xmean)/(std)
%   Output:
%       DATAout.
%           input = denormalized matrix             [p x N]

%% INITIALIZATIONS

option = OPTION.norm;   % gets normalization option from structure
X_norm = DATAin.input;	% gets data matrix from structure [p x N]

[p,N] = size(X_norm); 	% number of samples and attributes
Xmin = DATAin.Xmin;     % minimum value of each attribute
Xmax = DATAin.Xmax;     % maximum value of each attribute
Xmed = DATAin.Xmed;     % mean of each attribute
dp = DATAin.dp;         % standard deviation of each attribute

%% ALGORITHM

X = zeros(p,N); % initialize data

switch option
    case (1)    % denormalize between [0 e 1]
        for i = 1:p,
            for j = 1:N,
                X(i,j) = X_norm(i,j)*(Xmax(i) - Xmin(i)) + Xmin(i); 
            end
        end
    case (2)    % denormalize between [-1 e +1]
        for i = 1:p,
            for j = 1:N,
                X(i,j) = 0.5*(X_norm(i,j) + 1)*(Xmax(i) - Xmin(i)) + Xmin(i); 
            end
        end
    case (3)    % denormalize by the mean and standard deviation
        for i = 1:p,
            for j = 1:N,
                X(i,j) = X_norm(i,j)*dp(i) + Xmed(i);
            end
        end
    otherwise
        X = X_norm;
        disp('Choose a correct option. Data was not normalized.')
end

%% FILL OUTPUT STRUCTURE

DATAin.input = X;

DATAout = DATAin;

%% END