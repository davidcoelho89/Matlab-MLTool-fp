function [OUT] = lms_predict(DATA,PAR)

% --- LMS Regression Prediction ---
%
%   [OUT] = lms_predict(DATA,PAR)
%
%   Input:
%       DATA.
%           input = inputs matrix                       [p x N]
%       PAR.
%           W = transformation matrix                [No x p+1] or [No x p]
%           add_Bias = whether or not to add the bias   [0 or 1]
%   Output:
%       OUT.
%           y_h = estimated outputs matrix              [No x N]

%% ALGORITHM

[OUT] = linear_predict(DATA,PAR);

%% END