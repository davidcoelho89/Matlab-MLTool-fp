function [f_value] = get_harmonic(DATA,PAR,harm)
% Test
%

%% INITIALIZATION

% Get Data
Vsig = DATA.input;      % Spectrum Vector (Nfreq x 1)

% Get Parameters
Fs = PAR.Fs;            % Sampling Frequence (Hz)
T = PAR.T;              % Total Aquisition time (s)
win = PAR.win;          % Search for harmonic (Hz) between [-win/2 +win/2]

% Calculated Parameters
dt = 1/Fs;              % Sampling Period (s)
L = T*Fs;               % Signal Length
t = ((0:L-1)*dt)';      % Time vector

% Parameters for fft
Nfft = 2^nextpow2(L);               % number (power of 2) higher than signal length
Nfreq = Nfft/2 + 1;                 % discretization of frequencies
f = (Fs/2)*linspace(0,1,Nfreq)';    % Frequencies vector

%% ALGORITHM

% Search for Main frequency

f_main_max = 70*Nfreq/(Fs/2);       % Calculate number of pts for 70 Hz
f_main_max = floor(f_main_max);     % int number 

f_main = 0;
f_value = 0;

for i = 2:f_main_max,              	% Don't get CC signal -> Vsig(1)
   if Vsig(i) > f_value,
       f_main = i;
       f_value = Vsig(i);
   end
end

% Search for harmonic

search = Nfreq*(win/2)/(Fs/2);  % Search for harmonic (Hz) between [-win/2 +win/2]
search = floor(search);         % int number 

f_min = harm*f_main - search;   % min frequency
f_min = floor(f_min);           % int number

f_max = harm*f_main + search;   % max frequency
f_max = floor(f_max);           % int number

f_value = 0;

for i = f_min:f_max,
    if Vsig(i) > f_value,
        f_value = Vsig(i);
    end
end

%% FILL OUTPUT STRUCTURE

%% END