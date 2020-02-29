function DATAout = klda_transform(DATAin,PAR)

% --- Kernel Linear Discriminant Analysis Data Transformation ---
%
%   DATAout = klda_transform(DATAin,PAR)
%
%   Input:
%       DATAin.
%           input = input matrix                            [p x Nin]
%           output = output matrix                          [Nc x Nin]
%       PAR.
%           rem = mean removal [0 or 1]                     [cte]
%           mu = mean of input matrix                       [p x 1]
%           W = Transformation Coefficients                 [Ntr x q]
%           X = Transformation Vectors                      [p x Ntr]
%   Output:
%       DATAout.
%           input = input matrix                            [q x Nin]
%           output = output matrix                          [Nc x Nin]

%% INITIALIZATIONS

% Get Data
Xin = DATAin.input;     % input matrix
Y = DATAin.output;      % output matrix
[~,Nin] = size(Xin); 	% dimension of input matrix

% Get Parameters
rem = PAR.rem;          % mean removal
mu = PAR.mu;            % mean of training data set
W = PAR.W;              % Transformation matrix
[Ntr,q] = size(W);      % Dimension of transformation matrix
X = PAR.X;              % Samples used to transform data

%% ALGORITHM

% Remove mean

if (rem == 1),
    Xin = Xin - repmat(mu,1,Nin);
end

% Transform input matrix

Xtr = zeros(q,Nin);

for n = 1:Nin,
    xn = Xin(:,n);
    for k = 1:q,
        ak = W(:,k);
        for i = 1:Ntr,
            xi = X(:,i);
            Xtr(k,n) = Xtr(k,n) + ak(i)*kernel_func(xn,xi,PAR);
        end
    end
end

%% FILL OUTPUT STRUCTURE

DATAout.input = Xtr;
DATAout.output = Y;

%% END