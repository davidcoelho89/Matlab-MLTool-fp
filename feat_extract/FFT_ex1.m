%% FFT EXAMPLE

clear;
clc;
% clf;
format long g;

%% INIT

% Given Parameters
Fs = 10000;             % Sampling Frequence (Hz)
T = 10;                 % Total Aquisition time (s)
win = 10;               % Search for frequency (Hz) between [-win/2 +win/2]

% Calculated Parameters
dt = 1/Fs;              % Sampling Period (s)
L = T*Fs;               % Signal Length
t = ((0:L-1)*dt)';      % Time vector

% Parameters for fft
Nfft = 2^nextpow2(L);               % number (power of 2) higher than signal length
Nfreq = Nfft/2 + 1;                 % discretization of frequencies
f = (Fs/2)*linspace(0,1,Nfreq)';    % frequencies for spectrum plot

% At this sample:
%
% (Fs/2 = 5000 Hz)
% (Nfreq = Nfft/2 + 1 = 65537 pts)
%
% 5000Hz --> 65537 pts
% X Hz   --> Y pts

%% CREATE SIGNAL

f1 = 50;                % 50Hz (main frequency)

w1 = 2*pi*f1;           % first harmonic
a1 = 10;                % first amplitude
w2 = 2*w1;              %   "       "
a2 = 3;                 %   "       "
w3 = 3*w1;              %   "       "
a3 = 2;                 %   "       "

y1 = a1*sin(w1*t);      % first signal
y2 = a2*sin(w2*t);      % second signal
y3 = a3*sin(w3*t);      % third signal

y_res = (y1+y2+y3);     % result signal

figure; plot(t,y1,'r',t,y2,'b',t,y3,'g')

figure; plot(t,y_res);  % plot signal

%% FFT ALGORITHM

M1 = fft(y_res,Nfft)/L;         % Frequencies (with real and imaginary)
M2 = 2*abs(M1(1:Nfreq,:));      % Frequencies (module - columms)
arg_M2 = angle(M1(1:Nfreq,:));	% Frequencies (arguments)

figure; plot(f,M2);

%% RECONSTRUCT SIGNAL

y_h = zeros(L,1);

for i = 1:Nfreq,
    y_h = y_h + M2(i)*sin(2*pi*f(i)*t + arg_M2(i));
end

figure; plot(t,y_h);

%% GET HARMONICS

DATA.input = M2;

PAR.Fs = Fs;
PAR.T = T;
PAR.win = win;

harm1 = get_harmonic(DATA,PAR,1);
harm2 = get_harmonic(DATA,PAR,2);
harm3 = get_harmonic(DATA,PAR,3);

%% END