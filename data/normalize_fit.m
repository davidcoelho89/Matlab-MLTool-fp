function [PAR] = normalize_fit(DATA,OPTION)

% --- Generate Parameters for Normalization ---
%
%   [PAR] = normalize_fit(DATA,OPTION)
%
%   Input:
%       DATA.
%           input = data matrix                     [p x N]
%       OPTION.
%           norm = how will be the normalization
%               0: input_out = input_in
%               1: normalize between [0, 1]
%               2: normalize between [-1, +1]
%               3: normalize by z-score transformation (standardzation)
%                  (empirical mean = 0 and standard deviation = 1)
%                  Xnorm = (X-Xmean)/(std)
%   Output:
%       PAR.
%           norm = how will be the normalization    [cte]
%           Xmin = Minimum value of attributes      [p x 1]
%           Xmax = Maximum value of attributes      [p x 1]
%           Xmed = Mean value of attributes         [p x 1]
%           Xstd = Standard Deviation of attributes  [p x 1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(OPTION))),
    OPTION.norm = 3;
else
    if (~(isfield(OPTION,'norm'))),
        OPTION.norm = 3;
    end
end

%% ALGORITHM

% Get normalization option and data matrix [N x p]

normalization_option = OPTION.norm;
X = DATA.input';

% Get min, max, mean and standard deviation measures

Xmin = min(X)';
Xmax = max(X)';
Xmed = mean(X)';
Xstd = std(X)';

%% FILL OUTPUT STRUCTURE

PAR.norm = normalization_option;
PAR.Xmin = Xmin;
PAR.Xmax = Xmax;
PAR.Xmed = Xmed;
PAR.Xstd = Xstd;

%% END