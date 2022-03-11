% exemplo de filtragem medias moveis de sinal ruidoso
% Mediana das amostras mais recentes
% Autor: Guilherme A. Barreto
% Data: 19/11/2021

clear; 
clc; 
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Carrega audio gravado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x Fs] = audioread('ana1.wav');
N = length(x); 		% numero de amostras no sinal
T = N*(1/Fs);  		% Duracao do sinal em segundos
t = 0:1/Fs:T-1/Fs;	% Instantes de amostragem
sig = 0.5; 			% desvio-padrao do ruido gaussiano (media zero)

% gera sinal ruidoso
xn = x + sig*randn(size(x));

% Normaliza sinal filtrado para faixa [-1,+1]
xn = 2*((xn-min(xn))/(max(xn)-min(xn)))-1;   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gera senoide com ruido
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

##Fs=22050;  % frequencia de amostragem (Hz ou SPS - samples per second)
##Nbits=16;  % Numero de bits de resolucao
##
##T=3; % Duracao do sinal em segundos
##t=0:1/Fs:T-1/Fs;   % Instantes de amostragem
##
##F0=880;  % Frequencia fundamental (nota La um oitava acima 2*440 Hz)
##
##x=sin(2*pi*F0*t);  % Gera forma de onda senoidal pura (i.e. sem ruido)
##xn=x+sig*randn(size(x));  % gera sinal ruidoso

##xn=2*((xn-min(xn))/(max(xn)-min(xn)))-1;   % Normaliza sinal filtrado para faixa [-1,+1]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Filtro Media Moveis (media das M amostras mais recentes)    %%%%
%%%% xf1: sinal filtrado após 1 passada do filtro                %%%%
%%%% xf2: sinal filtrado após 2 passadas do filtro               %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = 10;
disp('Filtragem Mediana Movel - Passada 1');
xf1 = xn(1:L);  % Valores iniciais do sinal filtrado
for n = L:length(xn), 
   xf1(n) = median(xn(n-L+1:n));
end

% Normaliza sinal filtrado para faixa [-1,+1]
xf1 = 2*((xf1-min(xf1))/(max(xf1)-min(xf1)))-1;   

disp('Filtragem Mediana Movel - Passada 2');
xf2 = xf1(1:L);  % Valores iniciais do sinal filtrado
for n = L:length(xf1), 
   xf2(n) = median(xf1(n-L+1:n));
end

% Normaliza sinal filtrado para faixa [-1,+1]
xf2 = 2*((xf2-min(xf2))/(max(xf2)-min(xf2)))-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Graficos e Audios  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mostra primeiros ciclos do sinal
figure; 
subplot(2,2,1); 
plot(t(1:200), x(1:200));  
title('SINAL SEM RUIDO'); 
grid

% mostra primeiros ciclos do sinal ruidoso
subplot(2,2,2); 
plot(t(1:200), xn(1:200));  
title('SINAL COM RUIDO GAUSSIANO'); 
grid

% mostra primeiros ciclos do sinal filtrado 1
subplot(2,2,3); 
plot(t(1:200), xf1(1:200));  
title('SINAL FILTRADO (PASSADA 1)'); 
grid

% mostra primeiros ciclos do sinal filtrado 2
subplot(2,2,4); 
plot(t(1:200), xf2(1:200));  
title('SINAL FILTRADO (PASSADA 2)'); 
grid

% Toca o sinal livre de ruido
disp('Audio sinal SEM ruido');
sound(x,Fs);   
pause 

% Toca sinal ruidoso
disp('Audio sinal COM ruido');
sound(xn,Fs)  
pause

% Toca Sinal primeira filtrada
disp('Sinal 1a. Filtrada');
sound(xf1,Fs);  
pause

% Toca Sinal segunda filtrada
disp('Sinal 2a. Filtrada');
sound(xf2,Fs);