function [Kxy] = kernelFunction(x,y,model)

%
%  --- HELP about kernelFunction ---
%

%% SET DEFAULT HYPERPARAMETERS

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



%% ALGORITHM











Kxy = x + y + model;


