function Xmods = powerSpectralDensity(signals,Fs)

% --- Build attributes matrix using FFT method ---
%
%   Xfreq = powerSpectralDensity(signals,Fs);
%
%   Input:
%       signals = Cell with all singals     [Ns x 1]
%       Fs = Sampling Frequency             [cte] (Hz)
%   Output:
%       Xfreq = Attributes Matrix           [Ns x Nquad*p]

%% INIT

Nsignals = length(signals);

lim_faixa = [100,200,360,500,700,850,1000,1200, ...
                 1320, 1500, 2200, 2700, 3700];

Nfaixas = length(lim_faixa);

Xmods = zeros(Nsignals,Nfaixas);
Xfreq = zeros(Nsignals,Nfaixas);

%% ALGORITHM

for i = 1:Nsignals

    signal = signals{i};                % Get signal
    Ns = length(signal);                % Signal length
    Nfft = 2^nextpow2(Ns);              % Number (power of 2) > Ns
    Nfreq = Nfft/2 + 1;                 % Discretization of frequencies
    f = (Fs/2)*linspace(0,1,Nfreq)';	% frequencies for spectrum plot
    Xs = fft(signal,Nfft)/Ns;           % Components (real and imag parts)
    Xs_mod = 2*abs(Xs(1:Nfreq));        % Get Components' modules
    
    % plot frequencies (debug)
    if (i == 1)
        figure;
        plot(f,Xs_mod,'b-')
    end
    
    % Find maximum components in each frequency range
    cont = 1;
    for j = 1:Nfaixas

        lim_sup = lim_faixa(j);  % upper limit of the frequency range
        f_max = f(cont);         % Component value with maximum modulus
        Xs_max = Xs_mod(cont);   % Maximum module value
        
        while 1
            cont = cont+1;
            f_atual = f(cont);
            Xs_atual = Xs_mod(cont);
            
            if (Xs_atual > Xs_max)
                f_max = f_atual;
                Xs_max = Xs_atual;
            end
           
            if (f_atual > lim_sup)
                break;
            end
            
        end
        
        Xmods(i,j) = Xs_max;
        Xfreq(i,j) = f_max;
    end

end

% Debug => Verifica Densidade Espectral de Potencia
% figure;
% plot(f,Xs_mod,'b-')
% title('Densidade Espectral de Potência')
% xlabel('Frequências')
% ylabel('Amplitudes')

%% END