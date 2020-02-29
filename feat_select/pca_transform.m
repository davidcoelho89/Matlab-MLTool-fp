function DATAout = pca_transform(DATAin,PAR)

% --- Principal Component Analysis Data Transformation ---
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
Xin = DATAin.input;     % input matrix
Y = DATAin.output;      % output matrix
[~,Nin] = size(Xin);	% dimension of input matrix

% Get Parameters
rem = PAR.rem;          % mean removal
mu = PAR.mu;            % mean of training data set
W = PAR.W;              % Transformation matrix

%% ALGORITHM

% Remove mean from input matrix

if (rem == 1),
    Xin = Xin - repmat(mu,1,Nin);
end

% Transform input matrix (generate matrix -> [q x N])

Xtr = W' * Xin;

%% FILL OUTPUT STRUCTURE

DATAout.input = Xtr;
DATAout.output = Y;

%% END