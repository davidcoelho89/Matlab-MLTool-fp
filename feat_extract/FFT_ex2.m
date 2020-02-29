%% FFT
% David Nascimento coelho
% Última modificação: 31/01/12

clear;
clc;
format long g;

%% Complex Numbers

% z = 3 - 7*i;                  % complex number
% M = abs(z)                    % magnitude
% Ph = angle(z)                 % phase angle
% Ph2 = atan2(imag(z),real(z))  % phase angle

%% Espectro de Frequencias

Fs = 10000;                    % Frequencia de amostragem (10 KHz)
T = 1/Fs;                      % Perído da amostragem
L = 100000;                    % Largura do sinal (100.000 pontos)
t = (0:L-1)*T;                 % Vetor de tempo total

% Can comment this line
L = 2^nextpow2(L);             % Next power of 2 from length of data

% Matrix with fft in complex mode
Matriz_Freq = fft(data,L)/L;
f = Fs/2*linspace(0,1,L/2);
n = length(f);

% (deve "adicionar 1" à L/2 -> L/2+1, 
%  se mantiver linha do "nextpow2" descomentada)

% Module of Phase 1
Fase1 = 2*abs(Matriz_Freq(1:L/2+1,1));
% Module of Phase 2
Fase2 = 2*abs(Matriz_Freq(1:L/2+1,2));
% Module of Phase 3
Fase3 = 2*abs(Matriz_Freq(1:L/2+1,3));
% Module of Bobbin
Bobina = 2*abs(Matriz_Freq(1:L/2+1,7));

%% Plotagem dos Espectros

% Plot do espectro de frequencia da fase 1.
figure(5)
plot(f(1:n),Fase1(1:n))
grid
title('Espectro de frequencia de i1(t)')
xlabel('Frequencia (Hz)')
ylabel('|I1(f)|')
% AXIS([-10 500 -0.1 1.5])

% Plot do espectro de frequencia da fase 2.
figure(6)
plot(f(1:n),Fase2(1:n))
grid
title('Espectro de frequencia de i2(t)')
xlabel('Frequencia (Hz)')
ylabel('|I2(f)|')

% Plot do espectro de frequencia da fase 3.
figure(7)
plot(f(1:n),Fase3(1:n))
grid
title('Espectro de frequencia de i3(t)')
xlabel('Frequencia (Hz)')
ylabel('|I3(f)|')

% Plot do espectro de frequencia da bobina interna.
figure(8)
plot(f(1:n),Bobina(1:n))
grid
title('Espectro de frequencia de bobina(t)')
xlabel('Frequencia (Hz)')
ylabel('|BOBINA(f)|')

%% END
