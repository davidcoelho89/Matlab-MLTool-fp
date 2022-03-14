%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Exemplo de estimacao de parametros de  %%%%
%%% um modelo ARX(2,1) usando o método OLS %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; 
clc; 
%close all;

sig2 = 0.01;    % Variancia do ruido branco
N = 1000;       % No de amostras das series {u(k),y(k)}, k=1, ..., N

% Ordens da regressao de entrada e saida
n = 2; 
m = 1; 

%%%%%% Geracao das series de entrada (u) e de saida (y)

% SINAL DE ENTRADA (onda quadrada aleatoria)
u = round(rand(2*N,1));  

% SINAL DE SAIDA (sistema ARX)
y = zeros(1,n);
for k = n+1:2*N
  y(k) = 0.43*y(k-1)-0.67*y(k-2) + 1.98*u(k-1) + sqrt(sig2)*randn; 	
end

% Dados sem outliers
u = u(N+1:end); 
y = y(N+1:end);

% Dados com outliers
OUT = 1;        % Adiciona outlier (ruido impulsivo)
if OUT
   Pout = 0.05; % Porcentagem de contaminacao por outliers
   Nout = ceil(Pout*N);
   I = randperm(N);
   Iout = I(1:Nout);
   y(Iout) = 5*y(Iout);
end

%load actuator.mat; y=p;
%load exchanger.mat; y=y2; u=u2;

figure; 
stairs(u(1:100),'linewidth',2);

figure; 
plot(y,'linewidth',2);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ETAPA DE ESTIMACAO %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ordens supostas para fins de estimacao
n = 2; 
m = 1;

p = []; 
X = [];
N = length(y);
for k = max(n,m)+1:N
  % ARX(2,1)
  p = [p; y(k)]; 
  X = [X; y(k-1) y(k-2) u(k-1)]; 
end

B1 = pinv(X)*p;  % Estimacao OLS

% Predicao (saida vetorial ou todos de uma vez)

yhat = X*B1;  % Predicao da variavel de saida (com dados de estimacao)
residuos = p - yhat;

% Analise dos residuos modelo ARX-OLS

TAUmax = 100;                           % Max. lag para FAC
fcac = myfac3(residuos,TAUmax);         % estimativas da FCAC
limconf = (2/sqrt(N))*ones(1,TAUmax);

figure; 
stem(fcac,'linewidth',2)
hold on
plot(limconf,'r-','linewidth',2)
plot(-limconf,'r-','linewidth',2)
hold off

figure; 
histfit(residuos,20)

%% Estimacao via algoritmos LMS e LMM (robusto)

Kout = 0.3;         % Limiar de outlier
lr = 0.1;           % Passo de aprendizagem
Nr = 10;            % Numero de repeticoes (opcional)

w = zeros(n+m,1);   % Coeficientes lms
wr = zeros(n+m,1);  % Coeficientes lmm

% for i=1:Nr,
for k = max(n,m)+1:N
    x = [y(k-1); y(k-2); u(k-1)];
        
    % LMS convencional
    yhat3 = w'*x;
    erro_lms = y(k) - yhat3; 
    %w = w + lr*erro_lms*x;         % LMS usual
    w = w + lr*erro_lms*x/(x'*x);   % LMS normalizado
    
    % LMS robusto (LMM - Least Mean M-estimate)
    yhat4 = wr'*x;
    erro_lmm = y(k) - yhat4; 
    
    % Aplica funcao de Huber (Testa se erro = outlier)
    if abs(erro_lmm) < Kout
      q_lmm = 1;                    % se erro = normal
    else
      q_lmm = Kout/abs(erro_lmm);   % se erro = outlier
    end
      
    %wr=wr+lr*q_lmm*erro_lmm*x;   % LMM usual (nao normalizado)
    wr=wr+lr*q_lmm*erro_lmm*x/(x'*x);   % LMM normalizado
    
end
% end

% Rotina para calculo dos residuos dos modelos robustos  
for k = max(n,m)+1:N
    x = [y(k-1); y(k-2); u(k-1)];
        
    % LMS convencional
    yhat3 = w'*x;
    residuos_lms(k) = y(k) - yhat3; 
       
    % LMS robusto (LMM - Least Mean M-estimate)
    yhat4 = wr'*x;
    residuos_lmm(k) = y(k) - yhat4; 
end

param = [0.43; -0.67; 1.98];
disp(param);
disp([B1 w wr]);  % OLS LMS LMM
disp([norm(param-B1) norm(param-w) norm(param-wr)])

fcac_lms = myfac3(residuos_lms,TAUmax);  % FCAC_h para modelo ARX-LMS
fcac_lmm = myfac3(residuos_lmm,TAUmax);  % FCAC_h para modelo ARX-LMM

%%%%%%%%%%%%%%%%%%%%%
%%%%% FIGURAS  %%%%%%
%%%%%%%%%%%%%%%%%%%%%

figure; 
subplot(2,1,1); 
stem(fcac_lms,'linewidth',2); 
hold on;
title('FAC residuos (modelo ARX-LMS)');
plot(limconf,'r-','linewidth',2)
plot(-limconf,'r-','linewidth',2) 
set(gca, "fontsize", 14); 
hold off

subplot(2,1,2); 
stem(fcac_lmm,'linewidth',2);
hold on
title('FAC residuos (modelo ARX-LMM)');
plot(limconf,'r-','linewidth',2)
plot(-limconf,'r-','linewidth',2); 
set(gca, "fontsize", 14); 
hold off

figure; 
subplot(2,1,1); histfit(residuos_lms,20); 
title('Histograma Residuos (modelo ARX-LMS)');
set(gca, "fontsize", 14);
subplot(2,1,2); 
histfit(residuos_lmm,20); 
title('Histograma Residuos (modelo ARX-LMM)');
set(gca, "fontsize", 14);

%% END
