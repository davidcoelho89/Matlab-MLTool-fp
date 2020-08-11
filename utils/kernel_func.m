function [Kxy] = kernel_func(x,y,PAR)

% --- Calculate The Dot Product Between Two Vectors in Feature Space ---
%
%   [kxy] = kernel_func(x,y,PAR)
%
%   Input:
%       x = vector in original (work) space                     [p x 1]
%       y = vector in original (work) space                     [p x 1]
%       PAR.
%           Ktype = kernel type                                 [cte]
%               1 -> Linear 
%               2 -> Gaussian (default)
%               3 -> Polynomial
%               4 -> Exponencial / Laplacian
%               5 -> Cauchy
%               6 -> Log
%               7 -> Sigmoid
%               8 -> Kmod
%           sig2n = kernel regularization parameter             [cte]
%           sigma   (gauss / exp / cauchy / log / kmod)         [cte]
%           gamma   (poly / log / Kmod)                         [cte]
%           alpha   (poly / sigmoid)                            [cte]
%           theta   (lin / poly / sigmoid)                      [cte]
%   Output:
%       Kxy = result of dot product in feature space            [cte]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 2) || (isempty(PAR))),
    PARaux.Ktype = 2;   	% Kernel Type (gaussian)
    PARaux.sigma = 0.1;   	% Kernel Std (gaussian)
    PAR = PARaux;
else
    % The default values are defined by the kernel type
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 2;
    end
    if (~(isfield(PAR,'sigma'))),
        if (PAR.Ktype == 2),
            PAR.sigma = 0.1;
        elseif (PAR.Ktype == 4),
            PAR.sigma = 0.01;
        elseif (PAR.Ktype == 5),
            PAR.sigma = 0.1;
        elseif (PAR.Ktype == 6),
            PAR.sigma = 0.1;
        elseif (PAR.Ktype == 8),
            PAR.sigma = 1.58;
        else
            PAR.sigma = 0.1;
        end
    end
    if (~(isfield(PAR,'gamma'))),
        if(PAR.Ktype == 3),
            PAR.gamma = 2;
        elseif (PAR.Ktype == 6),
            PAR.gamma = 2;
        elseif (PAR.Ktype == 8),
            PAR.gamma = 3.5;
        end
    end
    if (~(isfield(PAR,'alpha'))),
        if(PAR.Ktype == 3),
            PAR.alpha = 1;
        elseif (PAR.Ktype == 7),
            PAR.alpha = 0.1;
        end
    else
    end
    if (~(isfield(PAR,'theta'))),
        if(PAR.Ktype == 1),
            PAR.theta = 0;
        elseif (PAR.Ktype == 3),
            PAR.theta = 1;
        elseif (PAR.Ktype == 7),
            PAR.theta = 0.1;
        end
    end
end

%% INITIALIZATIONS

% Get parameters

Ktype = PAR.Ktype;      % Kernel type

% Get kernel especific parameters

if (Ktype == 1),        % Linear
    theta = PAR.theta;
elseif (Ktype == 2),    % Gaussian
    sigma = PAR.sigma;
elseif (Ktype == 3),    % Polynomial
    alpha = PAR.alpha;
    theta = PAR.theta;
    gamma = PAR.gamma;
elseif (Ktype == 4),    % Exponencial / Laplacian
    sigma = PAR.sigma;
elseif (Ktype == 5),    % Cauchy
    sigma = PAR.sigma;
elseif (Ktype == 6),    % Log
    sigma = PAR.sigma;
    gamma = PAR.gamma;
elseif (Ktype == 7),    % Sigmoid
	alpha = PAR.alpha;
    theta = PAR.theta;
elseif (Ktype == 8),    % Kmod
    sigma = PAR.sigma;
    gamma = PAR.gamma;
    a = 1/(exp(gamma/sigma^2)-1);
end

%% ALGORITHM

if (Ktype == 1),        % Linear
    Kxy = (x' * y + theta);
elseif (Ktype == 2),    % Gaussian
    Kxy = exp(-norm(x-y)^2/(sigma^2));
elseif (Ktype == 3),    % Polynomial
    Kxy = (alpha * x' * y + theta)^gamma;
elseif (Ktype == 4),    % Exponencial / Laplacian
    Kxy = exp(-norm(x-y)/sigma);
elseif (Ktype == 5),    % Cauchy
    Kxy = (1 + (norm(x-y)^2)/(sigma^2))^(-1);
elseif (Ktype == 6),    % Log
    Kxy = -log(1 + (norm(x-y)^gamma)/(sigma^2));
elseif (Ktype == 7),    % Sigmoid (hyperbolic tangent)
    Kxy = tanh(alpha * x' * y + theta);
elseif (Ktype == 8),    % Kmod
    Kxy = a*(exp(gamma/(norm(x-y)^2+sigma^2))-1);
else                    % Use dot product if a wrong option was chosen
    Kxy = (x' * y);
end

%% END