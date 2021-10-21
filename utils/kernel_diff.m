function [Kdiff] = kernel_diff(x,w,PAR)

% --- Derivative Measure of kernel distance between two vectors  ---
%
%   [d] = kernel_diff(x,w,PAR)
% 
%   Input:
%       x = vector in original (work) space                     [p x 1]
%       w = vector in original (work) space                     [p x 1]
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
%       Kdiff = Derivative measure of kernel distance           [p x 1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 2) || (isempty(PAR))) 
    PARaux.Ktype = 2;   	% Kernel Type (gaussian)
    PARaux.sigma = 0.1;   	% Kernel Std (gaussian)
    PAR = PARaux;
else
    % The default values are defined by the kernel type
    if (~(isfield(PAR,'Ktype'))) 
        PAR.Ktype = 2;
    end
    if (~(isfield(PAR,'sigma'))) 
        if (PAR.Ktype == 2) 
            PAR.sigma = 0.1;
        elseif (PAR.Ktype == 4) 
            PAR.sigma = 0.01;
        elseif (PAR.Ktype == 5) 
            PAR.sigma = 0.1;
        elseif (PAR.Ktype == 6) 
            PAR.sigma = 0.1;
        elseif (PAR.Ktype == 8) 
            PAR.sigma = 1.58;
        else
            PAR.sigma = 0.1;
        end
    end
    if (~(isfield(PAR,'gamma'))) 
        if(PAR.Ktype == 3) 
            PAR.gamma = 2;
        elseif (PAR.Ktype == 6) 
            PAR.gamma = 2;
        elseif (PAR.Ktype == 8) 
            PAR.gamma = 3.5;
        end
    end
    if (~(isfield(PAR,'alpha'))) 
        if(PAR.Ktype == 3) 
            PAR.alpha = 1;
        elseif (PAR.Ktype == 7) 
            PAR.alpha = 0.1;
        end
    else
    end
    if (~(isfield(PAR,'theta'))) 
        if(PAR.Ktype == 1) 
            PAR.theta = 0;
        elseif (PAR.Ktype == 3) 
            PAR.theta = 1;
        elseif (PAR.Ktype == 7) 
            PAR.theta = 0.1;
        end
    end
end

%% INITIALIZATIONS

% Get parameters

Ktype = PAR.Ktype;      % Kernel type

% Get kernel especific parameters

if (Ktype == 1)         % Linear
    theta = PAR.theta;
elseif (Ktype == 2)     % Gaussian
    sigma = PAR.sigma;
elseif (Ktype == 3)     % Polynomial
    alpha = PAR.alpha;
    theta = PAR.theta;
    gamma = PAR.gamma;
elseif (Ktype == 4)     % Exponencial / Laplacian
    sigma = PAR.sigma;
elseif (Ktype == 5)     % Cauchy
    sigma = PAR.sigma;
elseif (Ktype == 6)     % Log
    sigma = PAR.sigma;
    gamma = PAR.gamma;
elseif (Ktype == 7)     % Sigmoid
	alpha = PAR.alpha;
    theta = PAR.theta;
elseif (Ktype == 8)     % Kmod
    sigma = PAR.sigma;
    gamma = PAR.gamma;
    a = 1/(exp(gamma/sigma^2)-1);
end

%% ALGORITHM

if (Ktype == 1)         % Linear
    Kdiff = 2*(x - w);
elseif (Ktype == 2)     % Gaussian
    Kdiff = (2/(sigma^2))*exp(-(norm(x-w)^2)/(2*(sigma^2)))*(x - w);
elseif (Ktype == 3)     % Polynomial
    Kdiff = 2*alpha*gamma*(x*(alpha*(w'*x)+theta)^(gamma-1) - ...
                           w*(alpha*(w'*w)+theta)^(gamma-1));
elseif (Ktype == 4)     % Exponencial / Laplacian
    Kdiff = (2/(sigma*norm(x-w)))*exp(-norm(x-w)/sigma)*(x-w);
elseif (Ktype == 5)     % Cauchy
    Kdiff = (4*(sigma^2))*(x-w)/((sigma^2 + norm(x-w)^2)^2);
elseif (Ktype == 6)     % Log
    Kdiff = 4*(x-w)/(sigma^2 + norm(x-w)^2);
elseif (Ktype == 7)     % Sigmoid
    Kdiff = 2*alpha*((x-w) - ...
              (x*tanh(alpha*(w'*x)+theta)^2 - w*tanh(alpha*(w'*w)+theta)^2));
elseif (Ktype == 8)     % Kmod
    Kdiff = 4*a*gamma*exp(gamma/(norm(x-w)^2+sigma^2))*(x-w) / ...
            (norm(x-w)^2+sigma^2)^2;
else                    % Use dot product if a wrong option was chosen
    Kdiff = 2*(x - w);
end

%% END