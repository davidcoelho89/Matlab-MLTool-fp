function [A] = ex_struct(B)
%
% FUNCAO TESTE
% Retorna a soma e a diferenca de dados de uma estrutura.
%

%% ALGORITMO

x = B.x;
y = B.y;

A.sum = x+y;
A.sub = x-y;

%% END