function [model] = initializeSysIdModel(model_name)

% --- Initialize a System Identification Model ---
%
%   [model] = initializeSysIdModel(model_name)
%
%   Input:
%       model_name = which classifier will be used
%           'lms'

%% ALGORITHM

modelString = strcat(model_name,'Arx');

model = feval(str2func(modelString));

%% END