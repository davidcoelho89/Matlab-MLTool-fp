function input = build_input_ts(OPTION)

% --- Generates an Input for System Identification Problems ---
%
%   input = build_input_ts(OPTION)
%
%   Input:
%       OPTION.input_type = type of input
%           'prbs': PseudoRandom Binary Signal
%       OPTION.input_length = length of signal
%   Output:
%       input = input time series [Ns x N]

%% SET DEFAULT OPTIONS

if(nargin == 0 || (isempty(OPTION)))
    OPTION.input_type = 'prbs';
    OPTION.input_length = 500;
end

%% INITIALIZATIONS

N = OPTION.input_length;

%% ALGORITHM

if(strcmp(OPTION.input_type,'prbs'))
    u_ts = round(rand(N,1));
end

%% FILL OUTPUT STRUCTURE

input = u_ts;

%% ENDss