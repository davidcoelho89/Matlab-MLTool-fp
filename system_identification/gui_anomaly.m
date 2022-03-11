%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Exemplo de estimacao de parametros de  %%%%
%%% um modelo ARX(2,1) usando o método OLS %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

pkg load signal
pkg load statistics

%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% ETAPA DE ESTIMACAO
%%%%%%%%%%%%%%%%%%%%%

sig2=0.01;  % variancia do ruido branco
N=500;   % num. de observacoes das series {u(k),y(k)}, k=1, ..., N

n=2; m=1; % Ordens da regressao de entrada e saida

%%%%%% Geracao das series de entrada (u) e de saida (y)
p=0.6; 
for l=1:2*N, 
  if rand<p, 
    u(l)=1; 
  else 
    u(l)=0; 
  endif
end

y=zeros(1,n);
for k=n+1:2*N,
  y(k) = 0.43*y(k-1)-0.67*y(k-2) + 1.98*u(k-1) + sqrt(sig2)*randn; 	
endfor

u=u(N+1:end);
y=y(N+1:end);

%load actuator.mat; y=p;
%load exchanger.mat; y=y2; u=u2;

figure; stairs(u(1:100),'linewidth',2);
figure; plot(y,'linewidth',2);

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ETAPA DE ESTIMACAO %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

n=2; m=1;  % Ordens supostas para fins de estimacao

p=[]; X=[];
N=length(y);
for k=max(n,m)+1:N,
  p=[p; y(k)]; 
  X=[X; y(k-1) y(k-2) u(k-1)]; % ARX(2,1)
endfor

B1=pinv(X)*p;

% Predicao (saida vetorial ou todos de uma vez)
yhat=X*B1;  % Predicao da variavel de saida (com dados de estimacao)
residuos=p-yhat;

% Analise dos residuos
TAUmax=100;  % Max. lag para FAC
fcac=myfac3(residuos,TAUmax);  % estimativas da FCAC
limconf=(2/sqrt(N))*ones(1,TAUmax);
figure; stem(fcac,'linewidth',2); hold on;
plot(limconf,'r-','linewidth',2,-limconf,'r-','linewidth',2); hold off
[Rxy lags]=xcorr(u(n+1:end),residuos,TAUmax,'coeff');
Idx=find(lags>=0);
figure; stem(lags(Idx),Rxy(Idx),'linewidth',2); hold on;
plot(limconf,'r-','linewidth',2,-limconf,'r-','linewidth',2); hold off
figure; histfit(residuos,20);

%%%%%%%%%%%%%%%%%%%%
%%%%%%%% ETAPA DE DETECCAO DE ANOMALIAS
%%%%%%%%%%%%%%%%%%%%%

% Estisticas dos residuos
m=mean(residuos);
s=std(residuos);

L3i=m-3*s;   % Limite inferior do intervalo de normalidade
L3s=m+3*s;   % Limite superior do intervalo de normalidade

clear u; clear y;

%%%%%% Geracao das series de entrada (u) e de saida (y)
p=0.6; 
for l=1:2*N, 
  if rand<p, 
    u(l)=1; 
  else 
    u(l)=0; 
  endif
end

y=zeros(1,n);
for k=n+1:N,
  if k<250,
      y(k) = 0.43*y(k-1)-0.67*y(k-2) + 1.98*u(k-1) + sqrt(sig2)*randn; 
      yh(k) = B1(1)*y(k-1)+B1(2)*y(k-2) + B1(3)*u(k-1); 
  else
      y(k) = 5 + 0.43*y(k-1)-0.67*y(k-2) + 1.98*u(k-1) + sqrt(sig2)*randn;  
      yh(k) = B1(1)*y(k-1)+B1(2)*y(k-2) + B1(3)*u(k-1); 
  end
  
  erro(k)=y(k)-yh(k);  % Erro de predicao no instante k
  
  % Rotina para deteccao de anomalias
  if (erro(k)>=L3i) & (erro(k)<=L3s),
    normal(k)=1;
  else
    normal(k)=-1;
  endif
endfor

figure;
subplot(2,1,1); plot(y,'linewidth',2); 
subplot(2,1,2); stairs(normal,'r-','linewidth',2);

