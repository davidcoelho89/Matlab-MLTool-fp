function [DATAout] = normalize_transform(DATAin,PAR)

% --- Normalize the raw data ---
%
%   [DATAout] = normalize_transform(DATAin,PAR)
%
%   Input:
%       DATAin.
%           input = data matrix                     [p x N]
%       PAR.
%           Xmin = Minimum value of attributes      [p x 1]
%           Xmax = Maximum value of attributes      [p x 1]
%           Xmed = Mean value of attributes         [p x 1]
%           Xdp = Standard Deviation of attributes  [p x 1]
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

% Get normalization option and data matrix [N x p]

normalization_option = PAR.norm;   
X = DATAin.input';      

% Get number of samples and attributes

[N,p] = size(X);

% Get min, max, mean and standard deviation measures

Xmin = PAR.Xmin;
Xmax = PAR.Xmax;
Xmed = PAR.Xmed;
Xstd = PAR.Xstd;

%% ALGORITHM

X_norm = zeros(N,p); % initialize data

switch normalization_option
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
                X_norm(i,j) = (X(i,j) - Xmed(j))/Xstd(j); 
            end
        end
    otherwise
        X_norm = X;
        disp('Choose a correct option. Data was not normalized.')
end

X_norm = X_norm'; % transpose data for [p x N] pattern

%% FILL OUTPUT STRUCTURE

DATAout = DATAin;
DATAout.input = X_norm;

%% END