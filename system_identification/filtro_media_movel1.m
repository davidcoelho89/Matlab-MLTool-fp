% exemplo de filtragem medias movies de sinal ruidoso
% Media das L amostras mais recentes
% Autor: Guilherme A. Barreto
% Data: 19/11/2021

clear;
clc;
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Carrega audio gravado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x,Fs]=audioread('ana1.wav');
N=length(x); % numero de amostras no sinal
T=N*(1/Fs);  % Duracao do sinal em segundos
t=0:1/Fs:T-1/Fs;
sig=0.1; % desvio-padrao do ruido gaussiano (media zero)
xn=x+sig*randn(size(x));  % gera sinal ruidoso
xn=2*((xn-min(xn))/(max(xn)-min(xn)))-1;   % Normaliza sinal filtrado para faixa [-1,+1]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Gera senoide com ruido
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Fs=22050;  % frequencia de amostragem (Hz ou SPS - samples per second)
%Nbits=16;  % Numero de bits de resolucao
%
%T=3; % Duracao do sinal em segundos
%t=0:1/Fs:T-1/Fs;   % Instantes de amostragem
%
%F0=880;  % Frequencia fundamental (nota La um oitava acima 2*440 Hz)
%
%x=sin(2*pi*F0*t);  % Gera forma de onda senoidal pura (i.e. sem ruido)
%xn=x+sig*randn(size(x));  % gera sinal ruidoso

%xn=2*((xn-min(xn))/(max(xn)-min(xn)))-1;   % Normaliza sinal filtrado para faixa [-1,+1]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Filtro Media Moveis (media das M amostras mais recentes)    %%%%
%%%% xf1: sinal filtrado após 1 passada do filtro                %%%%
%%%% xf2: sinal filtrado após 2 passadas do filtro               %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L=10;
disp('Filtragem Media Movel - Passada 1');
xf1=xn(1:L);  % Valores iniciais do sinal filtrado
for n=L:length(xn)
   xf1(n) = mean(xn(n-L+1:n));
end

xf1=2*((xf1-min(xf1))/(max(xf1)-min(xf1)))-1;   % Normaliza sinal filtrado para faixa [-1,+1]

disp('Filtragem Media Movel - Passada 2');
xf2=xf1(1:L);  % Valores iniciais do sinal filtrado
for n=L:length(xf1)
   xf2(n) = mean(xf1(n-L+1:n));
end

xf2=2*((xf2-min(xf2))/(max(xf2)-min(xf2)))-1;   % Normaliza sinal filtrado para faixa [-1,+1]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Graficos e Audios  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure; 
subplot(2,2,1); plot(t(1:200), x(1:200));  % mostra primeiros ciclos do sinal
title('SINAL SEM RUIDO'); grid

subplot(2,2,2); plot(t(1:200), xn(1:200));  % mostra primeiros ciclos do sinal ruidoso
title('SINAL COM RUIDO GAUSSIANO'); grid

subplot(2,2,3); plot(t(1:200), xf1(1:200));  % mostra primeiros ciclos do sinal filtrado 1
title('SINAL FILTRADO (PASSADA 1)'); grid

subplot(2,2,4); plot(t(1:200), xf2(1:200));  % mostra primeiros ciclos do sinal filtrado 2
title('SINAL FILTRADO (PASSADA 2)'); grid

disp('Audio sinal SEM ruido');
sound(x,Fs);   % Toca o sinal livre de ruido
pause 

disp('Audio sinal COM ruido');
sound(xn,Fs)  % sinal ruidoso
pause

disp('Sinal 1a. Filtrada');
sound(xf1,Fs);  % Sinal primeira filtrada

pause
disp('Sinal 2a. Filtrada');
sound(xf2,Fs);  % Sinal segunda filtrada