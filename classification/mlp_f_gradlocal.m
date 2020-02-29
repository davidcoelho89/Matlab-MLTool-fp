function [Di] = mlp_f_gradlocal(Yi,option)

% --- Local Gradient of MLP ---
%
%   [Di] = mlp_f_gradlocal(Yi,option)
%
%   There is a minimum of 0.05 output so as not to paralyze the learning
%
%   input:
%       Yi = activation function result
%       option = type of activation function
%           = 1: Sigmoidal -> output: [0 e 1]
%           = 2: Hyperbolic Tangent -> output: [-1 e +1
%   output:
%       Yi = Derivate of activation function

%% ALGORITHM

switch option
    case (1)    % derivate of sigmoidal function
        Di = Yi.*(1 - Yi) + 0.05;
    case (2)    % derivate of hyperbolic tangent
        Di = 0.5*(1-Yi.^2) + 0.05;     
    otherwise
        Di = Yi;
        disp('Invalid local gradient option')
end

%% END