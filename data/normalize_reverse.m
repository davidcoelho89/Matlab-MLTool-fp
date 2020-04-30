function [DATAout] = normalize_reverse(DATAin,PAR)

% --- Reverse the normalized data ---
%
%   [DATAout] = normalize_reverse(DATAin,PAR)
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
%               1: denormalize between [0, 1]
%               2: denormalize between [-1, +1]
%               3: denormalize by z-score transformation
%                  Empirical Mean = 0 and Standard Deviation = 1)
%                  Xnorm = (X-Xmean)/(std)
%   Output:
%       DATAout.
%           input = denormalized matrix             [p x N]

%% INITIALIZATIONS

X_norm = DATAin.input;	% gets data matrix from structure [p x N]
[p,N] = size(X_norm); 	% number of samples and attributes

option = PAR.norm;      % gets normalization option from structure
Xmin = PAR.Xmin;        % minimum value of each attribute
Xmax = PAR.Xmax;        % maximum value of each attribute
Xmed = PAR.Xmed;        % mean value of each attribute
Xstd = PAR.Xstd;        % standard deviation of each attribute

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
                X(i,j) = X_norm(i,j)*Xstd(i) + Xmed(i);
            end
        end
    otherwise
        X = X_norm;
        disp('Choose a correct option. Data was not normalized.')
end

%% FILL OUTPUT STRUCTURE

DATAout = DATAin;
DATAout.input = X;

%% END