function [C] = convolucao(A, B)
% [C] = convolucao(A, B)
% Determina a convolução entre dois vetores (2 sinais no tempo)
%   entradas:
%       - A: sinal 01
%       - B: sinal 02
%   Saidas:
%       - C: Resultado da convolução

tam_A = length(A);  % tamanho do sinal A
tam_B = length(B);  % tamanho do sinal B
tam_C = max([tam_A+tam_B-1, tam_A, tam_B]); % sinal convoluido

C = zeros(1,tam_C); % inicializa o sinal C

for x = 1:tam_C,
    for j = 1:tam_C,

        ind_A = j;      % indice do vetor A que será multiplicado
        ind_B = x-j+1;  % indice do vetor B que será multiplicado
        
        if ((ind_A > 0) & (ind_A < tam_A) & (ind_B > 0) & (ind_B < tam_B)),
            C(x) = C(x) + A(ind_A)*B(ind_B);
        else
            continue;
        end
    end
end
