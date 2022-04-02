function PAR = normalizeTimeSeries_fit(DATA,OPT)

% --- Normalize Time Series to a Specific Range ---
%
%   PAR = normalizeTimeSeries_fit(DATA,OPT)
%
%   Input:
%       DATA.
%           input = Matrix with input time series    	[Nu x N]
%           output = Matrix with output time series    	[Ny x N]
%       OPT.
%           norm_type = how will be the normalization   [cte]
%               0: input_out = input_in
%               1: normalize between [0, 1]
%               2: normalize between [-1, +1]
%               3: normalize by z-score transformation (standardzation)
%                  (empirical mean = 0 and standard deviation = 1)
%                  Xnorm = (X-Xmean)/(std)
%               4: Xnorm = (X-Xmean)/(3*std)
%   Output:
%       PAR.
%           norm_type = how will be the normalization	[cte]
%           Umin = Minimum value of inputs              [Nu x 1]
%           Umax = Maximum value of inputs              [Nu x 1]
%           Umed = Mean value of inputs                 [Nu x 1]
%           Ustd = Standard Deviation of inputs         [Nu x 1]
%           Ymin = Minimum value of outputs             [Ny x 1]
%           Ymax = Maximum value of outputs             [Ny x 1]
%           Ymed = Mean value of outputs                [Ny x 1]
%           Ystd = Standard Deviation of outputs        [Ny x 1]

%% SET DEFAULT OPTIONS

if(nargin == 1 || (isempty(OPT)))
    OPT.norm_type = 1;
else
    if (~(isfield(OPT,'norm_type')))
        OPT.norm_type = 1;
    end
end

%% INIT

% Get normalization option
norm_type = OPT.norm_type;

%% ALGORITHM

if(isfield(DATA,'input'))
    U = DATA.input';
    Umin = min(U)';
    Umax = max(U)';
    Umed = mean(U)';
    Ustd = std(U)';
else
    Umin = [];
    Umax = [];
    Umed = [];
    Ustd = [];
end

if(isfield(DATA,'output'))
    Y = DATA.output';
    Ymin = min(Y)';
    Ymax = max(Y)';
    Ymed = mean(Y)';
    Ystd = std(Y)';
else
    Ymin = [];
    Ymax = [];
    Ymed = [];
    Ystd = [];
end

%% FILL OUTPUT STRUCTURE

PAR.norm_type = norm_type;
PAR.Umin = Umin;
PAR.Umax = Umax;
PAR.Umed = Umed;
PAR.Ustd = Ustd;
PAR.Ymin = Ymin;
PAR.Ymax = Ymax;
PAR.Ymed = Ymed;
PAR.Ystd = Ystd;

%% END