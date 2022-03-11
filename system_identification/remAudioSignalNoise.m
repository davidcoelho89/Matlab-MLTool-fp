function signals_out = remAudioSignalNoise(signals_in,L)

% --- Remove noise of Audio Signals ---
%
%   signal_out = remAudioSignalNoise(signal_in,L)
%
%   Input:
%       signal_in = Cell with all signals                   [Ns x 1]
%       L = Filter Order                                    [cte]
%
%   Output:
%       signal_out = Cell with all signals (noise removed)  [Ns x 1]

%% INIT

Nsignals = length(signals_in);
signals_out = cell(Nsignals,1);
offset = floor(L/2);

%% ALGORITHM

for i = 1:Nsignals
    xn = signals_in{i}; % signal with noise
    signal_length = length(xn);
    xf = xn(1:offset);  
    for n = (offset+1):(signal_length-offset)
        xf(n) = mean(xn(n-offset:n+offset));
    end
    signals_out{i} = xf;
end

%% END