%% FFT EXAMPLE 2

clear;
clc;
format long g;

%% CREATE SIGNAL

% Vetor de Tempo
Fs = 10000;
dt = 1/Fs;
T = 2;
t = 0:dt:T;
L = length(t);

% Inicializa sinal de saida
y_quad = zeros(1,L);
y_tri = zeros(1,L);

f1 = 400;
w1 = 2*pi*f1;

% Quantidade de componentes na aproximacao
J = 50;

for k = 0:J-1
    y_quad = y_quad + (4/((2*k+1)*pi))*sin((2*k+1)*w1*t);
end

for k = 1:J
    y_tri = y_tri + ((-1)^(k+1))*(2/(k*pi))*sin(k*w1*t);
end

% Mostra sinal resultante
figure;
plot(t,y_quad)
str = strcat('Sinal Resultante com ',int2str(J),' Componentes');
title(str)
xlabel('Tempo')
ylabel('Amplitude')

figure;
plot(t,y_tri)
str = strcat('Sinal Resultante com ',int2str(J),' Componentes');
title(str)
xlabel('Tempo')
ylabel('Amplitude')

%% SOUND

% sound(y_quad,Fs);
% sound(y_tri,Fs);

%% END