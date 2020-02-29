function DATAout = lda_transform(DATAin,PAR)

% --- Linear Discriminant Analysis Data Transformation ---
%
%   DATAout = pca_transform(DATAin,PAR)
%
%   Input:
%       DATAin.
%           input = input matrix                            [p x N]
%           output = output matrix                          [Nc x N]
%       PAR.
%           rem = mean removal [0 or 1]                     [cte]
%           mu = mean of input matrix                       [p x 1]
%           W = Transformation Matrix                       [p x q]
%   Output:
%       DATAout.
%           input = input matrix                            [q x N]
%           output = output matrix                          [Nc x N]

%% INITIALIZATIONS

% Get Data
X = DATAin.input;   % input matrix
Y = DATAin.output;  % output matrix
[~,N] = size(X);    % dimension of input matrix

% Get Parameters
rem = PAR.rem;      % mean removal
mu = PAR.mu;        % mean of training data set
W = PAR.W;          % Transformation matrix

%% ALGORITHM

% Remove mean
if (rem == 1),
    X = X - repmat(mu,1,N);
end

% Transform input matrix
X = W' * X;     % generate matrix -> [q x N]

%% FILL OUTPUT STRUCTURE

DATAout.input = X;
DATAout.output = Y;

%% END