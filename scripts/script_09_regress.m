%% REGRESSÃO - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

nfig = 0;       % contagem do numero de figuras

%% APROXIMAR CURVAS E MODELOS MATEMATICOS (polyfit x polyval)

x = 1:7;                            % entradas reais
y = [1.2 1.6 2.3 2.8 3.9 4.5 5.6];  % saidas reais

fpar = polyfit(x,y,1);      % encontra parametros para polinomio de ordem x
faj = polyval(fpar,x);      % gera função aproximada

nfig = nfig + 1; figure(nfig)       % proxima figura
plot(x,y,'+blue',x,faj,'black')     % plota curva aproxima e real

%% SIMPLIFICAR MODELO MATEMATICO

% simplify

%% END