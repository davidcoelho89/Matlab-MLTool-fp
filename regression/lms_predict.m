function [OUT] = lms_predict(DATA,PAR)

% --- LMS Regression  Test ---
%
%   [OUT] = lms_predict(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix [p x N]
%       PAR.
%           W = transformation matrix [Nc x p+1]
%   Output:
%       OUT.
%           y_h = classifier's output [Nc x N]

%% ALGORITHM

[OUT] = linear_predict(DATA,PAR);

%% END