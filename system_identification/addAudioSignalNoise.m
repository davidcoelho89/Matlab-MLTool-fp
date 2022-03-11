function signals_out = addAudioSignalNoise(signals_in,noise_var)

% --- Add noise to Audio Signals ---
%
%   signal_out = addAudioSignalNoise(signal_in,noise_var)
%
%   Input:
%       signal_in = Cell with all signals                   [Ns x 1]
%       noise_var = Variance of gaussian noise to be added  [cte]
%
%   Output:
%       signal_out = Cell with all signals (noise added)    [Ns x 1]

%% INIT

Nsignals = length(signals_in);
signals_out = cell(Nsignals,1);
noise_std = sqrt(noise_var);

%% ALGORITHM

for i = 1:Nsignals
    signal_in = signals_in{i};
    signal_size = size(signal_in);
    signals_out{i} = signal_in + noise_std*randn(signal_size);
end

%% END