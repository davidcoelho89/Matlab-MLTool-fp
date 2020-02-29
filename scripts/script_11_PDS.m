%% AMOSTRAGEM DE SINAIS

% David Coelho
% Ultima alteração: 13/11/2014

clear;
clc;
close;

%% GERA SENOIDE DE 2 kHz

f = 2000;               % frequncia
T = 1/f;                % periodo
tmin = 0;               % tempo inicial
tmax = 10*T;            % tempo final
d = (1/10)*T;           % discretização do vetor

t = tmin:d:tmax;        % vetor de tempo (discretização)
x = sin(2*pi*f*t);      % vetor seno -> sin(w*t)

% Gerar figura
subplot(311); plot(t,x);
title('Senoide de 2 kHz','Fontsize',14)
ylabel('Amplitude','Fontsize',14)
xlabel('Tempo (s)','Fontsize',14)

%% AMOSTRA SINAL A 20 kHz (10x taxa de amostragem)

f1 = 20000;
T1 = 1/f1;

t1 = tmin:T1:tmax;
x1 = sin(2*pi*f*t1);

subplot(312)
plot(t1,x1);
hold on
stem(t1,x1)
title ('Senoide de 2 kHz amostrada a 20 kHz','Fontsize',14)
ylabel('Amplitude','Fontsize',14)
xlabel('Tempo','Fontsize',14)

%% AMOSTRA SINAL A 3 kHz (1.5x taxa de amostragem)

f2 = 3000;
T2 = 1/f2;

t2 = tmin:T2:tmax;      % periodo em que se retirou as amostras
x2 = sin(2*pi*f*t2);    % senoide com valores amostrados

subplot(313); plot(t2,x2);
hold on
stem(t2,x2);
title ('Senoide de 2 kHz amostrada a 3 kHz','Fontsize',14)
ylabel('Amplitude','Fontsize',14)
xlabel('Tempo','Fontsize',14)

%% END
