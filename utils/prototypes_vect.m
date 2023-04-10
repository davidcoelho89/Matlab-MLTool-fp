function [Cout] = prototypes_vect(Cin)

% --- Convert a Tensor of prototypes into a Matrix  ---
%
%   [Cout] = prototypes_vect(Cin)
% 
%   Input:
%       Cin = prototypes' tensor            	[p x Nk(1) x ... x Nk(Nd)]
%   Output:
%       Cout = prototypes1 matrix               [p x mult(Nk)]

%% INITIALIZATION

Cin_size = size(Cin);       % vector with Cin size
Nd = length(Cin_size) - 1;  % number of dimensions

%% ALGORITHM

if(Nd == 1)
    Cout = Cin;
else
    k = prod(Cin_size(2:end));
    p = Cin_size(1);
    Cout = reshape(Cin,[p k]);
end

%% END