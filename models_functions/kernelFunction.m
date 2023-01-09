function [kxy] = kernelFunction(x,y,model)

%
%  --- HELP about kernelFunction ---
%

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
    kxy = (x' * y + model.theta);
elseif (strcmp(model.kernel_type,'gaussian'))
    kxy = exp(-norm(x-y)^2/(model.sigma^2));
elseif (strcmp(model.kernel_type,'polynomial'))
    kxy = (model.alpha * x' * y + model.theta)^model.gamma;
elseif (strcmp(model.kernel_type,'exponential'))
    kxy = exp(-norm(x-y)/model.sigma);
elseif (strcmp(model.kernel_type,'cauchy'))
    kxy = (1 + (norm(x-y)^2)/(model.sigma^2))^(-1);
elseif (strcmp(model.kernel_type,'log'))
    kxy = -log(1 + (norm(x-y)^model.gamma)/(model.sigma^2));
elseif (strcmp(model.kernel_type,'sigmoid'))
    kxy = tanh(model.alpha * x' * y + model.theta);
elseif (strcmp(model.kernel_type,'kmod'))
    kxy = a*(exp(model.gamma/(norm(x-y)^2+model.sigma^2))-1);
else % Use dot product if a wrong option was chosen
    kxy = (x' * y);
end

%% END