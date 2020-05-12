% CELLS AND STRUCTS - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

%% CELLS

aux = [1 2 ; 3 4];  % matriz de teste

out{1} = aux;       % coloca em uma posição da celula

out{2} = aux+1;     % coloca em outra posicao da celula

out{2}(1,2) = 7;    % acessando a matriz de dentro da celula

%% STRUCTS

B.x = aux;
B.y = aux+1;

x = B.x;
y = B.y;

A.sum = x+y;
A.sub = x-y;

%% END