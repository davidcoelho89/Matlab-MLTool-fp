function [out] = ex_cell()

% TESTE DA CONSTRUCAO DE UMA CELULA EM MATLAB

aux = [1 2 ; 3 4];  % matriz de teste

out{1} = aux;       % coloca em uma posição da celula

out{2} = aux+1;     % coloca em outra posicao da celula

out{2}(1,2) = 7;    % acessando a matriz de dentro da celula

end

