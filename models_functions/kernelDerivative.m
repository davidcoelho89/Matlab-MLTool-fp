function [kder] = kernelDerivative(x,w,model)

% --- Derivative Measure of kernel distance between two vectors  ---
%
%   [kder] = kernelDerivative(x,w,PAR)
% 
%   Input:
%       x = vector in original (work) space                     [p x 1]
%       w = vector in original (work) space                     [p x 1]
%       model.
%           kernel_type =
%               'linear'
%               'gaussian (default)'
%               'polynomial'
%               'exponencial (laplacian)'
%               'cauchy'
%               'log'
%               'sigmoid'
%               'kmod'
%           regularization                                      [cte]
%           sigma   (gauss / exp / cauchy / log / kmod)         [cte]
%           gamma   (poly / log / Kmod)                         [cte]
%           alpha   (poly / sigmoid)                            [cte]
%           theta   (lin / poly / sigmoid)                      [cte]
%   Output:
%       Kdiff = Derivative measure of kernel distance           [p x 1]

%% SET DEFAULT HYPERPARAMETERS

if (nargin == 2)
    model.kernel_type = 2;
    model.sigma = 0.1;
else
    if(~isprop(model,'kernel_type'))
        model.kernel_type = 'gaussian';
    end
    if(~isprop(model,'sigma'))
        if (strcmp(model.kernel_type,'gaussian'))
            model.sigma = 0.1;
        elseif (strcmp(model.kernel_type,'exponential'))
            model.sigma = 0.01;
        elseif (strcmp(model.kernel_type,'cauchy'))
            model.sigma = 0.1;
        elseif (strcmp(model.kernel_type,'log'))
            model.sigma = 0.1;
        elseif (strcmp(model.kernel_type,'kmod'))
            model.sigma = 1.58;
        else
            model.sigma = 0.1;
        end
    end
    if (~isprop(model,'gamma'))
        if (strcmp(model.kernel_type,'polynomial'))
            model.gamma = 2;
        elseif (strcmp(model.kernel_type,'log'))
            model.gamma = 2;
        elseif (strcmp(model.kernel_type,'kmod'))
            model.gamma = 3.5;
        end
    end
    if (~isprop(model,'alpha'))
        if (strcmp(model.kernel_type,'polynomial'))
            model.alpha = 1;
        elseif (strcmp(model.kernel_type,'sigmoid'))
            model.alpha = 0.1;
        end
    end
    if (~isprop(model,'theta'))
        if (strcmp(model.kernel_type,'linear'))
            model.theta = 0;
        elseif (strcmp(model.kernel_type,'polynomial'))
            model.theta = 1;
        elseif (strcmp(model.kernel_type,'sigmoid'))
            model.theta = 0.1;
        end
    end
end

%% ALGORITHM

if (strcmp(model.kernel_type,'linear'))
    theta = model.theta;
    kder = 2*(x - w) + theta;
elseif (strcmp(model.kernel_type,'gaussian'))
    sigma = model.sigma;
    kder = (2/(sigma^2))*exp(-(norm(x-w)^2)/(2*(sigma^2)))*(x - w);
elseif (strcmp(model.kernel_type,'polynomial'))
    alpha = model.alpha;
    theta = model.theta;
    gamma = model.gamma;
    kder = 2*alpha*gamma*(x*(alpha*(w'*x)+theta)^(gamma-1) - ...
                           w*(alpha*(w'*w)+theta)^(gamma-1));
elseif (strcmp(model.kernel_type,'exponential'))
    sigma = model.sigma;
    kder = (2/(sigma*norm(x-w)))*exp(-norm(x-w)/sigma)*(x-w);
elseif (strcmp(model.kernel_type,'cauchy'))
    sigma = model.sigma;
    kder = (4*(sigma^2))*(x-w)/((sigma^2 + norm(x-w)^2)^2);
elseif (strcmp(model.kernel_type,'log'))
    sigma = model.sigma;
    %gamma = model.gamma;
    kder = 4*(x-w)/(sigma^2 + norm(x-w)^2);
elseif (strcmp(model.kernel_type,'sigmoid'))
	alpha = model.alpha;
    theta = model.theta;
    kder = 2*alpha*((x-w) - ...
              (x*tanh(alpha*(w'*x)+theta)^2 - w*tanh(alpha*(w'*w)+theta)^2));
elseif (strcmp(model.kernel_type,'kmod'))
    sigma = model.sigma;
    gamma = model.gamma;
    a = 1/(exp(gamma/sigma^2)-1);
    kder = 4*a*gamma*exp(gamma/(norm(x-w)^2+sigma^2))*(x-w) / ...
            (norm(x-w)^2+sigma^2)^2;
else % Use dot product if a wrong option was chosen
    theta = model.theta;
    kder = 2*(x - w) + theta;
end

%% END