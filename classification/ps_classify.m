function [OUT] = ps_classify(DATA,PAR)

% --- PS classifier test ---
%
%   [OUT] = ps_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix   [p x N]
%       PAR.
%           W = transformation matrix   [Nc x p+1]
%   Output:
%       OUT.
%           y_h = classifier's output   [Nc x N]

%% ALGORITHM

[OUT] = linear_classify(DATA,PAR);

%% END