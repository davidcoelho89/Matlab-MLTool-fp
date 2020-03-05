function DATAout = lda_transform(DATAin,PAR)

% --- Linear Discriminant Analysis Data Transformation ---
%
%   DATAout = lda_transform(DATAin,PAR)
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

%% FILL OUTPUT STRUCTURE

DATAout = pca_transform(DATAin,PAR);

%% END