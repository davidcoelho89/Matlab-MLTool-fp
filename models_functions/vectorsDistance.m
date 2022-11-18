function [d] = vectorsDistance(x,y,model)

%
%  --- HELP about vectorsDistance ---
%

%% SET DEFAULT HYPERPARAMETERS

if (nargin == 2)
    model.distance_measure = 2;
    model.kernel_type = 'none';
else
    if(~isprop(model,'distance_measure'))
        model.distance_measure = 2;
    end
    if(~isprop(model,'kernel_type'))
        model.kernel_type = 'none';
    end
end
    
%% ALGORITHM

if(strcmp(model.kernel_type,'none'))
    
    if (model.distance_measure == 0)            % Dot product
        d = (x')*y;
    elseif (model.distance_measure == 1)        % Manhattam (City-block)
        d = sum(abs(x - y));
    elseif (model.distance_measure == 2)        % Euclidean
        d = sqrt(sum((x - y).^2));
    elseif (model.distance_measure == inf)      % Maximum Minkowski (Chebyshev)
        d = max(abs(x - y));
    elseif (model.distance_measure == -inf)     % Minimum Minkowski
        d = min(abs(x - y));
    elseif (model.distance_measure > 2)         % Minkowski Distance
        d = sum(abs(x - y)^dist)^(1/dist);
    else
        d = sqrt(sum((x - y).^2));              % Euclidean as default
    end
    
else
    
    d = kernelFunction(x,x,model) - ...
        2*kernelFunction(x,y,model) + ...
        kernelFunction(y,y,model);
    
end

end