function [d] = vectors_dist(x,y,PAR)

% --- Measure distance between two vectors  ---
%
%   [d] = vectors_dist(x,y,PAR)
% 
%   Input:
%       x = sample vector 1                                     [p x 1]
%       y = sample vector 2                                     [p x 1]
%       PAR.
%           dist = type of distance                             [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance  
%               2: Euclidean distance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       d = distance between prototype and sample               [cte]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 2) || (isempty(PAR))),
    PARaux.dist = 2;
    PARaux.Ktype = 0;
    PAR = PARaux;
else
    if (~(isfield(PAR,'dist'))),
        PAR.dist = 2;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 0;
    end
end
    

%% INITIALIZATION

Ktype = PAR.Ktype;        	% Kernel type
dist = PAR.dist;        	% Choose distance

%% ALGORITHM

% Calculate distance for non-kernelized algorithms

if(Ktype == 0),
    
    if (dist == 0),                 % Dot product
        d = (x')*y;
    elseif (dist == 1),             % Manhattam (City-block)
        d = sum(abs(x - y));
    elseif (dist == 2),             % Euclidean
        d = sqrt(sum((x - y).^2));
    elseif (dist == inf),           % Maximum Minkowski (Chebyshev)
        d = max(abs(x - y));
    elseif (dist == -inf),          % Minimum Minkowski
        d = min(abs(x - y));
    elseif (dist > 2),              % Minkowski Distance
        d = sum(abs(x - y)^dist)^(1/dist);
    else
        d = sqrt(sum((x - y).^2));  % Euclidean as default
    end
    
% Calculate distance for kernelized algorithms

else
    
    d = kernel_func(x,x,PAR) - ...
        2*kernel_func(x,y,PAR) + ...
        kernel_func(y,y,PAR);

end

%% END