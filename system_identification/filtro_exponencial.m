% exemplo de filtragem passa-baixa de sinal ruidoso
% Suavizacao exponencial
% Autor: Guilherme A. Barreto
% Data: 19/11/2021

clear; 
clc; 
close;

Fs = 22050;  % frequencia de amostragem (Hz ou SPS - samples per second)
Nbits = 16;  % Numero de bits de resolucao (opcional)

T = 3; 				% Duracao do sinal em segundos
t = 0:1/Fs:T-1/Fs;	% Instantes de amostragem

F0 = 880;  % Frequencia fundamental (nota La um oitava acima 2*440 Hz)

% Gera forma de onda senoidal pura (i.e. sem ruido)
x = sin(2*pi*F0*t);  

% Mostra primeiros ciclos do sinal
figure; 
plot(t(1:100), x(1:100),'linewidth',2); 
grid 
axis([0 t(100) -1.5 1.5]); 
xlabel('Tempo (s)'); 
ylabel('Amplitude');
h = legend({"Sinal SEM RUIDO"},"location", "northeast");      
set(h, "fontsize", 12); set(gca, "fontsize", 14)

% Salva sinal em eps
print -deps -color 'semruido.eps'

% Toca o sinal livre de ruido
sound(x,Fs);   

sig = 0.1; 					% desvio-padrao do ruido gaussiano (media zero)
ruido = sig*randn(size(x)); % Gera ruido gaussiano de desvio-padrao sig

% Gera sinal ruidoso
xn = x + ruido;  

% Mostra primeiros ciclos do sinal ruidoso
figure; 
plot(t(1:100), xn(1:100),'linewidth',2); 
grid 
axis([0 t(100) -1.5 1.5]); xlabel('Tempo (s)'); ylabel('Amplitude');
h = legend({"Sinal COM RUIDO"},"location", "northeast");      
set(h, "fontsize", 12); set(gca, "fontsize", 14)

% Salva sinal em eps
print -deps -color 'comruido.eps'

% Toca o sinal ruidoso
sound(xn,Fs);

%[x Fs]=audioread('ana1.wav');
%N=length(x); % numero de amostras no sinal
%T=N*(1/Fs);  % Duracao do sinal em segundos
%t=0:1/Fs:T-1/Fs;
%sig=0.5; % desvio-padrao do ruido gaussiano (media zero)
%xn=x+sig*randn(size(x));  % gera sinal ruidoso
%xn=2*((xn-min(xn))/(max(xn)-min(xn)))-1;   % Normaliza sinal filtrado para faixa [-1,+1]
%
%figure; plot(t(1:100), x(1:100),'linewidth',2);  % mostra primeiros ciclos do sinal
%sound(x,Fs);   % Toca o sinal livre de ruido
%
%figure; plot(t(1:100), xn(1:100),'linewidth',2);  % mostra primeiros ciclos do sinal ruidoso
%sound(xn,Fs);   % Toca o sinal ruidoso

pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Filtro Passa-Baixa Discreto %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(xn);	% Numero de amostras do sinal ruidoso
a = 0.2;   		% Fator de suavizacao (smoothing)
xf = 0;    		% Inicializa sinal filtrado
for n = 2:N
  xf(n) = (1-a)*xf(n-1) + a*xn(n); 
end

% Normaliza sinal filtrado para faixa [-1,+1]
xf_norm=2*((xf-min(xf))/(max(xf)-min(xf)))-1;   

% Mostra primeiros ciclos do sinal filtrado
figure; 
plot(t(1:100), xf_norm(1:100),'linewidth',3); 
grid; 
axis([0 t(100) -1.5 1.5]); 
xlabel('Tempo (s)'); 
ylabel('Amplitude');
h = legend({"Sinal FILTRADO"},"location", "northeast");      
set(h, "fontsize", 12); 
set(gca, "fontsize", 14)

% Salva figura em EPS
print -deps -color 'filtrado1.eps'

% Toca o sinal filtrado
sound(xf_norm,Fs);   
