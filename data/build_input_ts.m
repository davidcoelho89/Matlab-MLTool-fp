function u_ts = build_input_ts(HP)

% --- Generates an Input for System Identification Problems ---
%
%   u_ts = build_input_ts(HP)
%
%   Input:
%       HP.
%           input_type = type of input
%               'awgn': Aditive White Gaussian Noise
%               'prbs': PseudoRandom Binary Signal
%               'aprbs': Amplitude modulated Pseudo-Random Binary Signal
%               'prbs_prob': prbs with "p" probability of being "1"
%       	input_length = length of signal
%           u_mean = mean of input signal (if needed)
%           u_var = variance of input signal (if needed)
%           prob = probability of signal being "1" (prbs_p)
%   Output:
%       input = input time series [Ns x N]

%% SET DEFAULT OPTIONS

if(nargin == 0 || (isempty(HP)))
    HP.input_type = 'awgn';     % What signal will be generated
    HP.input_length = 500;      % Signal length
    HP.u_mean = 0;              % Signal mean (if needed)
    HP.u_var = 0.5;             % Signal variance (if needed)
    HP.prob = 0.7;              % Probability of u(n) = 1 (if needed)
    
else
    if (~(isfield(HP,'input_type')))
        HP.input_type = 'awgn';
    end
    if (~(isfield(HP,'input_length')))
        HP.input_length = 500;
    end
    if (~(isfield(HP,'u_mean')))
        HP.u_mean = 0;
    end
    if (~(isfield(HP,'u_var')))
        HP.u_var = 0.5;
    end
    if (~(isfield(HP,'prob')))
        HP.prob = 0.5;
    end
    
end

%% INITIALIZATIONS

N = HP.input_length;
u_mean = HP.u_mean;
u_var = HP.u_var;
prob = HP.prob;

%% ALGORITHM

if(strcmp(HP.input_type,'awgn'))
    u_ts = u_mean + sqrt(u_var)*randn(N,1);

elseif(strcmp(HP.input_type,'prbs'))
    u_ts = round(rand(N,1));

elseif(strcmp(HP.input_type,'aprbs'))
    prbs_signal = round(rand(N,1));
    u_ts = zeros(N,1);
    u_ts(1) = rand;
    for n = 2:N
%         if(prbs_signal(n) == prbs_signal(n-1))
%             u_ts(n) = u_ts(n-1);
%         else
%             u_ts(n) = rand;
%         end
        if(prbs_signal(n) == 1 && prbs_signal(n-1) == 0)
            u_ts(n) = rand;
        else
            u_ts(n) = u_ts(n-1);
        end
    end
    
elseif(strcmp(HP.input_type,'prbs_prob'))
    u_ts = zeros(N,1);
    rand_signal = rand(N,1);
    for n = 1:N
        if(rand_signal(n) < prob)
            u_ts(n) = 1;
        else
            u_ts(n) = 0;
        end
    end
    
end

%% END