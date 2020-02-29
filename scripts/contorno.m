function [Vet_xy] = contorno(d,x0,y0)
% Determina as curvas de contorno (1, 3, 5) para uma determinada norma, dada
% as coordenadas centrais.
%   entradas:
%       - d: qual a norma -> 1 (city-block); 2 (euclidiana) ; 3 (chebyshev)
%       - x0: coordenada central do eixo x
%       - y0: coordenada central do eixo y
%   Saidas:
%       - Vet_xy = pontos pertencentes ao lugar geométrico

C = 1;  % Contorno 1
err = 0.001;
Vet_xy = [];

for i = -abs(C-x0):0.001:abs(C-x0),
    for j = -abs(C-y0):0.001:abs(C-y0),
        if d == 1,
            if (abs(i-x0)+abs(j-y0)) == C,
                Vet_xy = [Vet_xy; i j];
            end
        end
        if d == 2,
            if (((i-x0)^2+(j-y0)^2) < ((C^2)+err)) & ((((i-x0)^2+(j-y0)^2) > ((C^2)-err))) ,
                Vet_xy = [Vet_xy; i j];
            end
        end
        if d == 3,
            if (abs(i) == C) | (abs(j) == C),
               Vet_xy = [Vet_xy; i j];
            end
        end
    end
end
