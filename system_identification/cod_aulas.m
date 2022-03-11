%% IDENTIFICACAO DE SISTEMAS

% Implementacoes da Disciplina
% Autor: David Nascimento Coelho
% Data: 16/01/2022

close all;
clear;
clc;

%% Paths e testes de funções

% Get full path of current file
filePath = matlab.desktop.editor.getActiveFilename;
% fprintf('%s\n',filePath);

% Use the right symbol for the path                                                                           
if(isunix)   
    symb = '/';                   
else
    symb = '\'; 
end

% Find position of last "\" or "/"
nchar = length(filePath);
for n = 1:nchar
    if(filePath(n) == symb)
        lastSymb = n;
    end
end

% Folder path
folderPath = filePath(1:lastSymb);
% fprintf('%s\n',folderPath);

% Add folder and subfolders to the path
addpath(genpath(folderPath));

% Extra: Get the names of all present opened files
% openFiles = matlab.desktop.editor.getAll;
% fileNames = {openFiles.Filename};

% Clear variables
clear filePath folderPath i lastSymb nchar symb

%% PROCESSOS ESTOCASTICOS - GERA SINAIS

% Parametros iniciais

f0 = 440; 			% Freq. fundamental (Hz)
w0 = 2*pi*f0;		% Freq. fundamental (rad/s)

Fs = 10000; 		% Freq. amostragem (Hz)
dt = 1/Fs;          % Periodo de amostragem (s - distancia entre amostras)

T = 3; 				% Duracao do sinal (s)
t = 0:dt:(T-dt);    % Instantes de amostragem (vetor)

std_n = 0.1;    	% Desvio padrao do ruido (constante)

% Numero de amostras para mostrar primeiros ciclos de uma senoide

nciclos = 3;
nsamples = nciclos*ceil(Fs/f0);

% Senoide

xn = sin(w0*t);

figure; 
plot(t(1:nsamples),xn(1:nsamples),'r-','linewidth',1)

% Senoide + ruido branco

xr = xn + std_n*randn(size(t));	% Adiciona ruido ao sinal

figure; 
plot(t(1:nsamples),xr(1:nsamples),'r-','linewidth',1)

% Onda Quadrada

y = sign(xn);    % Verifica se valor é positivo ou negativo

figure; 
plot(t(1:nsamples),y(1:nsamples),'r-','linewidth',1)

% Onda Quadrada ruidosa

yr = y + std_n*randn(size(t));	% Adiciona ruido ao sinal

figure; 
plot(t(1:nsamples),yr(1:nsamples),'r-','linewidth',1)

% Sinal de amplitude constante (definida aleatoriamente)

A = unifrnd(-1,1);      % Amplitude do sinal (constante)
xn = A*ones(size(t));    % Sinal

figure; 
plot(t(1:nsamples),xn(1:nsamples),'r-','linewidth',1)

% Senoide com amplitude constante (definida aleatoriamente)

A = unifrnd(150,290);	% Amplitude da senoide (constante)
xn = A*sin(w0*t);        % Sinal

figure;
plot(t(1:nsamples),xn(1:nsamples),'r-','linewidth',1)

% Senoide com fase constante (definida aleatoriamente)

A = unifrnd(-pi,+pi);   % Fase constante (rad)
xn = sin(w0*t + A);      % Sinal

figure; 
plot(t(1:nsamples),xn(1:nsamples),'r-','linewidth',1)

%% PROCESSOS ESTOCASTICOS - MEDIDAS

% Processos Estocasticos (revisao) - Slide 39

Nr = 1000;       % Numero de realizações do experimento

% Senoide + ruido branco: Parametros

f0 = 60;                % Freq. fundamental (Hz)
w0 = 2*pi*f0;           % Freq. fundamental (rad/s)
Fs = 5000;              % Freq. amostragem (Hz)
dt = 1/Fs;              % Periodo de amostragem (s)
T = 3;                  % Duracao do sinal (s)
n = 0:dt:(T-dt);        % Instantes de amostragem (vetor)

noise_std = 0.5;     	% Desvio padrao do ruido (constante)

% Realizações

X = zeros(Nr,length(n));
for n = 1:Nr
    X(n,:) = sin(w0*n) + noise_std*randn(size(n));
end

% Medidas 1 (cada instante de tempo i - cada coluna - "ensemble")

mean_x = mean(X);   % Media (Ensemble mean)
var_x = var(X);     % Variancia (Ensemble variance)
%std_x = std(X);    % Desvio padrao (Ensemble standard deviation)
std_x = var_x.^0.5; % Desvio padrao (Ensemble standard deviation)
mad_x = mad(X);     % Desvio absoluto medio (Ensemble mean absolute dev)

% Grafico: Media +- Desvio Padrao

mi_up = mean_x + std_x;
mi_lw = mean_x - std_x;

figure; 
hold on
plot(n(1:140),mean_x(1:140),'r-','linewidth',2);
plot(n(1:140),mi_up(1:140),'b-','linewidth',2);
plot(n(1:140),mi_lw(1:140),'b-','linewidth',2);
hold off;

% Funcao de AutoCorrelacao  (FAC)  - Rx(t1,t2)
% Funcao de AutoCovariancia (FACV) - Cx(t1,t2)
% Funcao Coeficiente de AutoCorrelacao (FCAC) - px(t1,t2)

Rx = zeros(length(n),length(n));
Cx = zeros(length(n),length(n));
px = zeros(length(n),length(n));

for n = 1:length(n)
    disp(n);
    Xi = X(:,n);
    for j = n:length(n)
        Xj = X(:,j);
        
        Rx(n,j) = mean(Xi.*Xj);
        Rx(j,n) = Rx(n,j);
        
        %Cx(i,j) = mean( (Xi-mean_x(i)).*(Xj-mean_x(j)) );
        Cx(n,j) = Rx(n,j) - mean_x(n)*mean_x(j);
        Cx(j,n) = Cx(n,j);
        
        px(n,j) = Cx(n,j)/(std_x(n)*std_x(j));
        px(j,n) = px(n,j);

    end
end

% Valor Quadratico Medio

Rx_ii = diag(Rx);

%% SALVAR MATRIZES - Rx, Cx, px

filename = 'fac_facv_fcac.mat';

save(filename,'X','Rx','Cx','px');

%% ESTACIONARIEDADE / ERGODICIDADE  - VERIFICACAO 1

% Get FAC, FACV, FCAC from file
filename = 'fac_facv_fcac.mat';
load(filename);
clear filename;

% Get parameters
f0 = 60;                % Freq. fundamental (Hz)
% w0 = 2*pi*f0;         % Freq. fundamental (rad/s)
Fs = 5000;              % Freq. amostragem (Hz)
dt = 1/Fs;              % Periodo de amostragem (s)
T = 3;                  % Duracao do sinal (s)
t = 0:dt:(T-dt);        % Instantes de amostragem (vetor)

% Numero de amostras para mostrar primeiros ciclos de uma senoide

nciclos = 4;
nsamples = nciclos*ceil(Fs/f0);

% Get one realization and plot it

xn = X(1,:);
figure;
plot(t(1:nsamples),xn(1:nsamples),'r-');

% Get mean of each time
mean_x = mean(X);
figure;
plot(t(1:nsamples),mean_x(1:nsamples),'r-');

% Get variance of each time
var_x = var(X);
figure;
plot(t(1:nsamples),var_x(1:nsamples),'r-');

% Get Acf of each time
N = length(X(1,:));
for tau = 0:N-1
    px_tau = diag(px,tau)';
    t = 1:length(px_tau);
    figure;
    plot(t,px_tau,'r-');
    pause;
end

%% ESTACIONARIEDADE / ERGODICIDADE  - VERIFICACAO 2

% Build Signals

N0 = 50;
w0 = 2*pi/N0;
N2 = 20;
w2 = 2*pi/N2;
noise_std = 0.1;
noise_std2 = 0.3;
n = 1:1000;
n1 = 1:500;
n2 = 501:1000;

x1 = sin(w0*n) + noise_std*randn(size(n));

figure;
plot(n,x1,'b-');

x21 = sin(w0*n1) + noise_std*randn(size(n1));
x22 = sin(w0*n2) + 2 + noise_std*randn(size(n2));
x2 = [x21 x22];

figure;
plot(n,x2,'b-');

x31 = sin(w0*n1) + noise_std*randn(size(n1));
x32 = sin(w0*n2) + noise_std2*randn(size(n2));
x3 = [x31 x32];

figure;
plot(n,x3,'b-');

x41 = sin(w0*n1) + noise_std*randn(size(n1));
x42 = sin(w2*n1) + noise_std*randn(size(n1));
x4 = [x41 x42];

figure;
plot(n,x4,'b-');

% Verification parameters

nparts = 10;
tol = 0.11;
TAUmax = 50;

% Verify Signals` Stationarity

disp("Signal 1: ");
isStationary_x1 = stationarityTest(x1,nparts,tol,TAUmax);
disp("Signal 2: ");
isStationary_x2 = stationarityTest(x2,nparts,tol,TAUmax);
disp("Signal 3: ");
isStationary_x3 = stationarityTest(x3,nparts,tol,TAUmax);
disp("Signal 4: ");
isStationary_x4 = stationarityTest(x4,nparts,tol,TAUmax);

%% ESTACIONARIEDADE / ERGODICIDADE  - VERIFICACAO 3

isStationary_x1_2 = stationarityTest_gui(x1,nparts,tol);
isStationary_x2_2 = stationarityTest_gui(x2,nparts,tol);
isStationary_x3_2 = stationarityTest_gui(x3,nparts,tol);
isStationary_x4_2 = stationarityTest_gui(x4,nparts,tol);

%% FAC - RUIDO GAUSSIANO BRANCO

n = 1:1000;
std_n = 0.1;
TAUmax = 50;
tau = 0:TAUmax;
tau2 = -TAUmax:TAUmax;

vr = std_n*randn(size(n));

figure;
plot(n,vr,'b-');

tic; fac1 = acf(vr,TAUmax); t1 = toc;
tic; fac2 = accf(vr,TAUmax); t2 = toc;
tic; fac3 = acvf(vr,TAUmax); t3 = toc;
tic; fac4 = acvcf(vr,TAUmax); t4 = toc;
tic; fac5 = xcorr(vr,vr,TAUmax,'unbiased'); t5=toc;

tempos = [t1 t2 t3 t4 t5];
disp(tempos);

figure; stem(tau,fac1);
figure; stem(tau,fac2);
figure; stem(tau,fac3);
figure; stem(tau,fac4);
figure; stem(tau2,fac5);

%% FAC - SENOIDE + RUIDO

N = 50;         % periodo discreto
w0 = 2*pi/N;    % freq. discreta
Ns = 10000;     % numero de amostras
n = 0:Ns;       % vetor de amostras
std_n = 0.1;    % desvio padrao do ruido
TAUmax = 200;   % Maximo lag
tau = 0:TAUmax; % Vetor de lags

xn = cos(w0*n);
xr = xn + std_n*randn(size(n));

figure; 
plot(n,xn,'b-');

fac1 = acf(xn,TAUmax);
figure; stem(tau,fac1);

figure;
plot(n,xr,'b-');

fac2 = acf(xr,TAUmax);
figure; stem(tau,fac2);

%% FAC - MODELO AR(1)

% Parametros do Modelo

dp = 0.1;   % Desvio-padrao do ruido
a1 = 0.85;  % Coeficiente do modelo
N = 5000;   % Numero de pontos

% AR usando a funcao FILTER

xn = filter(1,[1 -a1],dp*randn(1,N)); 

% AR com loop for

% x(1) = dp*randn;  % valor inicial
% for n = 2:2*N
%   x(n) = a1*x(n-1) + dp*randn;
% end
% x = x(N+1:end);
%(descarta metade inicial (warm-up) para evitar influencia de cond inicial)

% Calculando as funções de autocorrelação

TAUmax = 50;

tic; fac1 = acf(xn,TAUmax); t1 = toc;
tic; fac2 = accf(xn,TAUmax); t2 = toc;
tic; fac3 = acvf(xn,TAUmax); t3 = toc;
tic; fac4 = acvcf(xn,TAUmax); t4 = toc;
tic; fac5 = xcorr(xn,xn,TAUmax,'unbiased'); t5=toc;

figure; stem(fac1);
figure; stem(fac2);
figure; stem(fac3);
figure; stem(fac4);
figure; stem(fac5);

%% MODELO AR(1) - ESTIMAÇÃO DOS PARAMETROS

% Parameters

a0 = 2;
a1 = 0.8;
s2e = 0.1;
TAU = 50;
tau = 0:50;

% Theorical Parameters

mux = a0/(1 - a1);
s2x = s2e/(1 - a1^2);
Rx = s2x*(a1.^(tau));

% Plot Theorical ACF

figure; 
stem(tau,Rx);
xlabel('Espaçamento temporal (lag)');
ylabel('FAC'); 
title('FAC teórica de um processo AR(1)');
axis([-1 51 -0.05 0.3]);

% Realization

N = 5500;       % Number of points
xn = zeros(N,1); % Random signal
xn(1) = rand;    % First sample
for t = 2:N
    xn(t) = a0 + a1*xn(t-1) + sqrt(s2e)*randn;
end
xn = xn(501:end); % forget first samples (warm-up)

% Plot signal

figure; 
plot(xn(1:200));
xlabel('Instante de Tempo');
ylabel('Amplitude');
title('Realizacao de um Processo AR(1)');

% Histogram

figure; 
histfit(xn);
title('Histograma das Amplitudes');

% Visualize ACF from Realization

Rx_h = xcov(xn,TAU,'unbiased');

figure; 
stem(tau,Rx_h(TAU+1:end));
xlabel('Espacamento temporal (lag)');
ylabel('FAC'); 
title('FAC empirica de um processo AR(1)');
axis([-1 51 -0.05 0.3]);

% Estimate Parameters

mux_h = mean(xn);
s2x_h = var(xn);
a1_h = Rx_h(TAU+2)/s2x; %Rx_h(2); 
a0_h = mux_h*(1-a1_h);
s2e_h = s2x_h*(1 - a1_h^2);

%% CROSS-CORRELATION FUNCTION

% Slide 104! Correlacao entre dois sinais (mesma dimensao)
% Rxy(tau) = E[X(t)Y(t+tau)] = Ryx(-tau)
% Ryx(tau) = E[Y(t)X(t+tau)] = Rxy(-tau)
% Para processos independentes: Rxy(tau) = Ryx(tau)
% Caso ergodicos: slide 109! estimação!

%% FILTRAGEM PASSA-BAIXA (Sinal Ruidoso)

% Gera Sinal

Fs = 22050; % Frequencia de amostragem (Hz ou SPS - samples per second)
dt = 1/Fs;  % Periodo de amostragem (s)
Nbits = 16; % Numero de bits de resolucao (de amplitude)

T = 5;              % Duracao do sinal em segundos
n = 0:dt:(T-dt);	% Instantes de amostragem (vetor)

f0 = 880;       % Frequencia fundamental (LA - 4a oitava - 2*440 Hz)
w0 = 2*pi*f0;   % Frequencia em rad/s
xn = sin(w0*n);  % Gera forma de onda senoidal pura (i.e. sem ruido)

figure; 
plot(n(1:100), xn(1:100));   % mostra primeiros ciclos do sinal

sound(xn,Fs,Nbits);          % toca o sinal livre de ruido

pause;

sig = 0.1;                          % desvio-padrao do ruido gaussiano
mu = 0;                             % media do ruido gaussiano
ruido = mu + sig*randn(size(xn));    % Gera ruido gaussiano

xn = xn + ruido;  % gera sinal ruidoso

figure; 
plot(n(1:100), xn(1:100));  % mostra primeiros ciclos do sinal ruidoso

sound(xn,Fs,Nbits);         % toca o sinal ruidoso

pause;

% Implementação do Filtro Passa-baixa discreto

N = length(xn);     % Numero de amostras do sinal ruidoso
a = 0.8;            % Fator de suavizacao (smoothing)

xf = zeros(N,1);
for n = 1:N-1
  xf(n+1) = a*xf(n) + (1-a)*xn(n); 
end

% Normaliza sinal filtrado para faixa [-1,+1]
xf_norm = 2*((xf-min(xf))/(max(xf)-min(xf)))-1;   

figure; 
plot(n(1:100), xf_norm(1:100));	% mostra primeiros ciclos do sinal filtrado

sound(xf_norm,Fs,Nbits);        % Toca o sinal filtrado

%% APROXIMACAO DE SINAIS POR SERIE DE FOURIER

T = 3;              % Duracao do sinal
f0 = 50;            % Frequencia fundamental da onda
Fs = 1500;          % Frequencia de amostragem
dt = 1/Fs;          % Distancia entre amostras
t = 0:dt:T;         % Instantes de amostragem

xn = sin(2*pi*f0*t); % Senoide pura
y = sign(xn);        % Onda quadrada resultante

% Plota onda quadrada e senoide na frequencia fundamental
figure;
hold on
plot(t(1:100),xn(1:100),'r-','linewidth',2);
plot(t(1:100),y(1:100),'b-','linewidth',2);
hold off
grid

% Aproximacao da onda quadrada por series de Fourier
Nh = 4;         % Numero de harmonicos
soma = 0;
for k = 1:Nh
  % Harmonicos impares
  aux = (1/(2*k-1))*sin(2*pi*(2*k-1)*f0*t);   
  soma = soma + (4/pi)*aux;
end

% Plota onda quadrada e onda reconstruida a partir dos harmonicos impares
figure;
hold on
plot(t(1:100),soma(1:100),'r-','linewidth',3);
plot(t(1:100),y(1:100),'b-','linewidth',3);
hold off
grid

% sound(y,Fs);    % Som da onda quadrada
% sound(soma,Fs); % Som da onda reconstruida a partir dos harmonicos

%% FFT



%% DENSIDADE ESPECTRAL DE POTENCIA

% Slide 122

%% LAG PLOTS



%% ESTIMADOR DE MAXIMA VEROSSIMILHANÇA



%% GERA PROCESSO AR(p)

% Define Parameters

sig2 = 0.15;     % Variancia do ruido
dp = sqrt(sig2); % Desvio-padrao do ruido
a = [0.3 0.6];   % Coeficientes de um processo AR(2): a1 = 0.3; a2 = 0.6
N = 10000;       % Numero de amostras
p = 2;           % Ordem do modelo AR(p)  
TAUmax = 50;     % Maximo lag da FAC

% y[n] = a1*y[n-1] + a2*y[n-2] + b1*u[n-1] + b2*u[n-2] + v[n]

% Gera vetor aleatorio
vn = dp*randn(1,N);

% AR usando a funcao FILTER do Octave/Matlab
ts = filter(1,[1 -a],vn);

% Figura contendo serie temporal
figure;
plot(ts);

% Figura contendo ACF
fac = acvcf(ts,TAUmax);
figure;
stem(fac);

% Figura contendo PACF
facp = pacf(ts,TAUmax);
figure;
stem(facp);

%% METODO DE YULE-WALKER ( Modelo AR(p) )

% Estima Parametros usando a funcao "aryule" do pacote "signal"
ah_aryule = aryule(ts,p);
ah_aryule = -ah_aryule(2:end)';

% Estimacao usando o metodo Yule-Walker (funcao feita)
ah_yw = ar_yw(ts,p);

%% GERA MATRIZ E VETOR PARA REGRESSAO

[X,y] = arxRegressionMatrixFromTS(ts,p);

N = length(y);

DATA.input = X';
DATA.output = y';

%% MINIMOS QUADRADOS

ah_MQ = (X'*X)\(X'*y);
%ah_MQ = inv(X'*X)*X'*vp;
%ah_MQ = pinv(X)*vp;

PAR = ols_train(DATA);
OUT = ols_classify(DATA,PAR);

%% LMS

eta = 0.005;
ah_lms = randn(2,1);

for n = 1:N
    xn = X(n,:)';
    yh = ah_lms' * xn;
    er = y(n) - yh;
    ah_lms = ah_lms + eta * er * xn;
end

%% Normalized LMS

eta = 0.005;
alpha = 0.05;
ah_nlms = randn(2,1);

for n = 1:N
    xn = X(n,:)';
    yh = ah_nlms' * xn;
    er = y(n) - yh;
    eta_norm = eta/(alpha + xn'*xn);
    ah_nlms = ah_nlms + eta_norm * er * xn;
end

%% Signal LMS

eta = 0.005;
ah_slms = randn(2,1);

for n = 1:N
    xn = X(n,:)';
    yh = ah_slms' * xn;
    er = y(n) - yh;
    ah_slms = ah_slms + eta * sign(er) * xn;
end

%% Leaky LMS

eta = 0.005;
lambda = 0.0001;
ah_llms = randn(2,1);

for n = 1:N
    xn = X(n,:)';
    yh = ah_llms' * xn;
    er = y(n) - yh;
    ah_llms = (1-lambda)*ah_llms + eta * er * xn;
end

%% Median LMS

eta = 0.005;
ah_mlms = randn(2,1);
m = 2;

errors = zeros(m,1);
for n = 1:m-1
    xn = X(n,:)';
    yh = ah_mlms' * xn;
    errors(n) = y(n) - yh;
end

for m = 1:N
    xn = X(n,:)';
    yh = ah_mlms' * xn;
    er = y(n) - yh;
    ah_mlms = ah_mlms + eta * er * xn;
end

%% RLS (1)



%% RLS (2)



%% IRLS



%% LMM



%% RLM



%% VALIDACAO DOS MODELOS (A partir dos residuos)

% Pela funcao aryule

yh1 = X*ah_aryule;          % Valores Preditos
res1 = y - yh1;             % Residuos
Pres1 = accf(res1,TAUmax);	% Coeficiente de Autocorrelação dos Residuos

figure; histfit(res1,40);
xlabel('residuos'); title('Histograma dos Residuos: AR(2) - aryule');
set(gca, "fontsize", 14)

figure; stem(Pres1,'k-','linewidth',2);
xlabel('lag'); title('FAC dos residuos: AR(2) - aryule');
set(gca, "fontsize", 14)

% Pelo metodo Yule-Walker

yh2 = X*ah_yw;              % Valores Preditos
res2 = y - yh2;             % Residuos
Pres2 = accf(res2,TAUmax);  % Coeficiente de Autocorrelação dos Residuos

figure; histfit(res2,40);
xlabel('residuos'); title('Histograma dos residuos: AR(2) - YW');
set(gca, "fontsize", 14)

figure; stem(Pres2,'k-','linewidth',2);
xlabel('lag'); title('FAC dos residuos: AR(2) - YW');
set(gca, "fontsize", 14)

% Pelo metodo Minimos-Quadrados 1

yh3 = X*ah_MQ;              % Valores Preditos
res3 = y - yh3;             % Residuos
Pres3 = accf(res3,TAUmax);  % Coeficiente de Autocorrelação dos Residuos

figure; histfit(res3,40);
xlabel('residuos'); title('Histograma dos residuos: AR(2) - MQ');
set(gca, "fontsize", 14)

figure; stem(Pres3,'k-','linewidth',2);
xlabel('lag'); title('FAC dos residuos: AR(2) - MQ');
set(gca, "fontsize", 14)

% Pelo metodo Minimos-Quadrados 2

yh4 = OUT.y_h';         	% Valores preditos
res4 = y - yh4;             % Residuos
Pres4 = accf(res4,TAUmax);	% Coeficiente de Autocorrelação dos Residuos

figure; histfit(res4,40);
xlabel('residuos'); title('Histograma dos residuos: AR(2) - MQ2');
set(gca, "fontsize", 14)

figure; stem(Pres4,'k-','linewidth',2);
xlabel('lag'); title('FAC dos residuos: AR(2) - MQ2');
set(gca, "fontsize", 14)

%% END