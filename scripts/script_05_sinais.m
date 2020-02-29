%% GRÁFICOS - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

nfig = 0;       % contagem do numero de figuras

%% Seno e cosseno 1 (função matlab e aproximação por series de potencia)

teta = 50;                  % angulo em graus
teta = (teta*pi)/180;       % angulo em radianos
aprox = 3;                  % quão boa é a aproximação

cos1 = 1;                   % cos em series de potencia
sen1 = teta;                % sin em series de potencia

for i = 1:aprox,
    cos1 = cos1 + ((-1)^i)*(teta^(2*i))/factorial(2*i);
    sen1 = sen1 + ((-1)^i)*(teta^(2*i+1))/factorial(2*i+1);
end

cos2 = cos(teta);
sen2 = sin(teta);

%% Seno e cosseno 2 - Curva

x = 0:0.01:10;
y = sin(x);
nfig = nfig + 1; figure(nfig)
plot (x,y)
axis([-0.1 10 -2 2])

%% Exponencial 1 (função matlab e aproximação por series de potencia)

x = 3;                      % valor de referencia
aprox = 10;                 % quão boa é a aproximação

Exp1 = exp(x);              % valor do matlab
Exp2 = 1;                   % exponencial em series de potencial

for i = 1:aprox,
    Exp2 = Exp2 + (x^i)/factorial(i);
end

%% Exponencial 2 - Curva

x = 0:0.01:10;
y = exp(-x);
nfig = nfig + 1; figure(nfig)
plot(x,y)
axis ([-1 5 -0.1 1.1])

%% Funcao Degrau

x = -5:0.01:5;
for i = 1:1001,
    if x(i) < 0,
        y(i) = 0;
    else
        y(i) = 1;
    end
end
nfig = nfig + 1; figure(nfig)
plot(x,y)
axis ([-3 3 -0.1 1.1])

%% Funcao Rampa

a = 2;
x = 0:0.01:10;
y = a*x;
nfig = nfig + 1; figure(nfig)
plot(x,y)
axis ([-0.1 5 -0.1 10])

%% Função translada

x = 0.001:0.001:10;
y = sin(x);
nfig = nfig + 1; figure(nfig)
plot(x,y)
axis ([-0.01 12 -1.1 1.1])

a = 0.001:0.001:1;
x1 = x+1;
b = zeros(1,1000);
x1 = [a x1];
y1 = [b y];

nfig = nfig + 1; figure(nfig)
plot(x1,y1)
axis ([-0.01 12 -1.1 1.1])

%% Soma de Funções

k = 2;              % angulo da reta
T = 5;              % tempo em que a reta fica constante
t = 0:0.01:10;      % tempo total
[~,tam] = size(t);  % quantos pontos de tempo

y1 = k*t;           % rampa1

y2 = k*(t-T);       % rampa2

for i = 1:tam,      % loop para indicar 
    if (t(i)-T) < 0,
        y2(i) = 0;
    end
end

y3 = y1-y2;         % função resultante

nfig = nfig + 1; figure(nfig)
plot(t,y3)
axis ([-1 11 -1 11])

nfig = nfig + 1; figure(nfig)
plot(t,y1)
axis ([-1 11 -1 21])

nfig = nfig + 1; figure(nfig)
plot(t,y2)
axis ([-1 11 -1 11])

%% DENTE DE SERRA E TRIANGULAR

t = 0:0.1:10;                   %vetor de tempo

y1 = sawtooth(t);               %dente de serra
nfig = nfig + 1; figure(nfig)
plot(t,y1)
axis ([-0.1 11 -1.1 1.1])

y2 = sawtooth(t,0.5);           %triangular
nfig = nfig + 1; figure(nfig)
plot(t,y2)
axis ([-0.1 11 -1.1 1.1])

%% ONDA QUADRADA

t = 0:0.1:10;                   %vetor de tempo

y1 = square(t);                 % onda quadrada simétrica
nfig = nfig + 1; figure(nfig)
plot(t,y1)
axis ([-0.1 11 -1.1 1.1])

y2 = square(t,20);              % porcentagem do periodo (duty cycle)
nfig = nfig + 1; figure(nfig)
plot(t,y2)
axis ([-0.1 11 -1.1 1.1])

%% SINAIS COM RUIDO

load BANCO_ESPEC_1
data2 = data1';

b = 0.00001; % 10% no máximo
a = -b;

r = a + (b-a).*rand(210,6);
r = r + [data2(1:42,:);data2(1:42,:);data2(1:42,:);data2(1:42,:);data2(1:42,:)];

rot2 = zeros(1,210);
data2 = r';

data3 = [data1(:,1:42) data2(:,:) data1(:,43:end)];
rot3 = [rot(:,1:42) rot2(:,:) rot(:,43:end)];

%% CONVOLUCAO ENTRE SINAIS (A)

deltaT = 0.01;          %discretização
t = 0:deltaT:10;        %vetor de tempo
h = zeros(1,length(t)); %inicializa sinal h
g = zeros(1,length(t)); %inicializa sinal g

for i=1:length(t),
    if (i > 400) & (i <= 600),
        h(i) = 1;
    end
    if (i > 500) & (i <= 600),
        g(i) = 2;
    end
end

C = convolucao(g,h)*deltaT; % vetor de convolução
tc = 1:length(C);           % vetor de tempo C
tc = tc*deltaT;             % discretização

figure;
subplot(2,2,1), plot(t,g);
axis([0 10 -3 3])
subplot(2,2,2), plot(t,h);
axis([0 10 -3 3])
subplot(2,2,3), plot(tc,C);
axis([5 15 -1 3])

%% CONVOLUCAO ENTRE SINAIS (B)

deltaT = 0.01;          %discretização
t = 0:deltaT:10;        %vetor de tempo
h = zeros(1,length(t)); %inicializa sinal h
g = zeros(1,length(t)); %inicializa sinal g

for i=1:length(t),
    if (i > 400) & (i <= 600),
        h(i) = 1;
    end
    if (i > 400) & (i <= 500),
        g(i) = -2;
    elseif (i > 500) & (i <= 600),
        g(i) = 2;
    end
end

C = convolucao(g,h)*deltaT; % vetor de convolução
tc = 1:length(C);           % vetor de tempo C
tc = tc*deltaT;             % discretização

figure;
subplot(2,2,1), plot(t,g);
axis([0 10 -3 3])
subplot(2,2,2), plot(t,h);
axis([0 10 -3 3])
subplot(2,2,3), plot(tc,C);
axis([5 15 -3 3])

%% CONVOLUCAO ENTRE SINAIS (C)

deltaT = 0.01;          %discretização
t = 0:deltaT:10;        %vetor de tempo
h = zeros(1,length(t)); %inicializa sinal h
g = zeros(1,length(t)); %inicializa sinal g

for i=1:length(t),
    if (i > 400) & (i <= 600),
        h(i) = 1;
    end
    if (i > 400) & (i <= 500),
        g(i) = (i-400)*deltaT;
    elseif (i > 500) & (i < 600),
        g(i) = g(i-1)-deltaT;
    end
end

C = convolucao(g,h)*deltaT; % vetor de convolução
tc = 1:length(C);           % vetor de tempo C
tc = tc*deltaT;             % discretização

figure;
subplot(2,2,1), plot(t,g);
axis([0 10 -3 3])
subplot(2,2,2), plot(t,h);
axis([0 10 -3 3])
subplot(2,2,3), plot(tc,C);
axis([5 15 -3 3])

%% CONVOLUCAO ENTRE SINAIS (D)

deltaT = 0.01;          %discretização
t = 0:deltaT:10;        %vetor de tempo
h = zeros(1,length(t)); %inicializa sinal h
g = zeros(1,length(t)); %inicializa sinal g

for i=1:length(t),
    if (i > 400) & (i <= 500),
        h(i) = (i-400)*deltaT;
    elseif (i > 500) & (i < 600),
        h(i) = h(i-1)-deltaT;
    end
    
    if (i > 400) & (i <= 500),
        g(i) = (i-400)*deltaT;
    elseif (i > 500) & (i < 600),
        g(i) = g(i-1)-deltaT;
    end
end

C = convolucao(g,h)*deltaT; % vetor de convolução
tc = 1:length(C);           % vetor de tempo C
tc = tc*deltaT;             % discretização

figure;
subplot(2,2,1), plot(t,g);
axis([0 10 -3 3])
subplot(2,2,2), plot(t,h);
axis([0 10 -3 3])
subplot(2,2,3), plot(tc,C);
axis([5 15 -3 3])

%% END