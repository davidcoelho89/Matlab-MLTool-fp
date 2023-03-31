function [model] = initializeClusteringModel(model_name)

% --- Initialize a Clustering Model ---
%
%   model = initializeClusteringModel(model_name)
%
%   Input:
%       model_name = which clustering algorithm will be used
%           'wta'

%% ALGORITHM

clusteringString = strcat(model_name,'Clustering');

model = feval(str2func(clusteringString));

%% END