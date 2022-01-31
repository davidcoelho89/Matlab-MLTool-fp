Fs = 10000;                    % Frequencia de amostragem
T = 1/Fs;                      % Tempo de amostragem
L = 100000;                    % Largura do sinal
t = (0:L-1)*T;                 % Vetor de tempo

% Next power of 2 from length of data
NFFT = 2^nextpow2(L); 
DATA = fft(data,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);

% Plot do espectro de frequencia da fase 1.
figure(1)
% plot(f,2*abs(Y(1:NFFT/2+1))) -> era o que tinha escrito no help
plot(f,2*abs(DATA(1:NFFT/2+1,1))) 
title('Espectro de frequencia de i1(t)')
xlabel('Frequencia (Hz)')
ylabel('|I1(f)|')

% Plot do espectro de frequencia da fase 2.
figure(2)
plot(f,2*abs(DATA(1:NFFT/2+1,2))) 
title('Espectro de frequencia de i2(t)')
xlabel('Frequencia (Hz)')
ylabel('|I2(f)|')

% Plot do espectro de frequencia da fase 3.
figure(3)
plot(f,2*abs(DATA(1:NFFT/2+1,3))) 
title('Espectro de frequencia de i3(t)')
xlabel('Frequencia (Hz)')
ylabel('|I3(f)|')

% Plot do espectro de frequencia da bobina interna.
figure(4)
plot(f,2*abs(DATA(1:NFFT/2+1,4))) 
title('Espectro de frequencia de bobina(t)')
xlabel('Frequencia (Hz)')
ylabel('|BOBINA(f)|')