function [Zi] = rbf_f_ativ(ui,ri,ativ)

% --- RBF Activation Function ---
%
%	[Zi] = rbf_f_ativ(Ui,ativ)
%
%   input:
%       ui = neuron output                      [cte]
%       ri = neuron radius                      [cte]
%       ativ = type of activation function      [cte]
%           1: gaussian 
%           2: multiquadratic
%           3: inverse multiquadratic 
%   Output:
%       Yi = result of activation function      [cte]

%% ALGORITHM

switch ativ
    case (1)    % Guassian
        Zi = exp(-(ui^2)/(2*(ri^2)));
    case (2)    % multiquadratic
        Zi = sqrt(ri^2 + ui^2);
    case (3)    % inverse multiquadratic
        Zi = 1/sqrt(ri^2 + ui^2);
    otherwise
        disp('Invalid activation function option')
end

%% END