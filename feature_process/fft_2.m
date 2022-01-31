% Frequencia de amostragem 10kH.
% Total de pontos 100.000

Fs = 10000;                    % Frequencia de amostragem
T = 1/Fs;                      % Tempo de amostragem
L = 100000;                    % Largura do sinal
t = (0:L-1)*T;                 % Vetor de tempo

% Gets directly the length of data
DATA = fft(data)/L;
f = Fs/2*linspace(0,1,L/2);

% Plot do espectro de frequencia da fase 1.
figure(1)
plot(f,2*abs(DATA(1:L/2,1)))
grid
title('Espectro de frequencia de i1(t)')
xlabel('Frequencia (Hz)')
ylabel('|I1(f)|')

% Plot do espectro de frequencia da fase 2.
figure(2)
plot(f,2*abs(DATA(1:L/2,2)))
grid
title('Espectro de frequencia de i2(t)')
xlabel('Frequencia (Hz)')
ylabel('|I2(f)|')

% Plot do espectro de frequencia da fase 3.
figure(3)
plot(f,2*abs(DATA(1:L/2,3)))
grid
title('Espectro de frequencia de i3(t)')
xlabel('Frequencia (Hz)')
ylabel('|I3(f)|')

% Plot do espectro de frequencia da bobina interna.
figure(4)
plot(f,2*abs(DATA(1:L/2,4)))
grid
title('Espectro de frequencia de bobina(t)')
xlabel('Frequencia (Hz)')
ylabel('|BOBINA(f)|')

% Plot do espectro de frequencia da fase 3 e da bobina interna.
%figure(5)
%plot(f,2*abs(DATA(1:L/2,4)),'r',f,2*abs(DATA(1:L/2,3)),'g')
%grid
%title('Espectro de frequencia de i3(t) e bobina(t)')
%xlabel('Frequencia (Hz)')
%ylabel('|BOBINA(f)|')