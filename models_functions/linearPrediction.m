function [yh] = linearPrediction(model,X)

% --- Linear Prediction (Classifiers or Regressors) ---
%
%   yh = linearPrediction(model,X)
%
%   Input:
%       model.
%           add_bias = add or not bias  [cte]
%           W = transformation matrix   [Nc x p+1] or [Nc x p]
%       X = attributes matrix           [p x N]
%   Output:
%       yh = classifier's output        [Nc x N]

%% ALGORITHM

[~,N] = size(X);
if(model.add_bias)
     X = [ones(1,N) ; X];
end
yh = model.W * X;

%% END