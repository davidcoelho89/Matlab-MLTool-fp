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

%% FILL OUTPUT STRUCTURE

DATAout = kpca_transform(DATAin,PAR);

%% END