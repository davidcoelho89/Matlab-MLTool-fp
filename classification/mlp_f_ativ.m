function [Yi] = mlp_f_ativ(Ui,option)

% --- MLP Classifier activation function ---
%
%	[Yi] = mlp_f_ativ(Ui,option)
%
%   input:
%       Ui = neuron activation
%       option = type of activation function
%           = 0: Linear -> output = input [-Inf +Inf]
%           = 1: Sigmoidal -> output: [0 1]
%           = 2: Hyperbolic Tangent -> output: [-1 +1]
%   Output:
%       Yi = output of non-linear function

%% ALGORITHM

switch option
    case (0)    % linear function => [-Inf,+Inf]
        Yi = Ui;
    case (1)    % sigmoidal function => [0,1]
        Yi = 1./(1+exp(-Ui));
    case (2)    % hyperbolic tangent function => [-1,+1]
        Yi = (1-exp(-Ui))./(1+exp(-Ui));
    otherwise
        Yi = Ui;
        disp('Invalid activation function option');
end

%% END