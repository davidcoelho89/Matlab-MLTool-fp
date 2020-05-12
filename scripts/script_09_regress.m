%% REGRESSÃO - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

nfig = 0;       % contagem do numero de figuras

%% APROXIMAR CURVAS E MODELOS MATEMATICOS (polyfit x polyval)

% Input and Outputs
x = 1:7;                            
y = [1.2 1.6 2.3 2.8 3.9 4.5 5.6];

% Calculates parameters of polynomial of degree d.
d = 1;
Ppar = polyfit(x,y,d);

% Generates values of polynomial Ppar evaluated at x.
yh = polyval(Ppar,x);

% Plot original and approximate curve
nfig = nfig + 1; figure(nfig)	
plot(x,y,'+blue',x,yh,'black')

%% SIMPLIFICAR MODELO MATEMATICO

% simplify

%% END