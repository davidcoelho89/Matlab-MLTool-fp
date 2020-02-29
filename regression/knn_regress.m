function [y_h,EQ2] = KNN_regress(k,P,T1,Q,T2)
%
% KNN Utilizado para regressão
% Ultima modificação: 23/10/2014
%
% [y_h,EQ] = KNN_regress(k,P,T1,Q,T2)
%
% O resultado desta é a estimação da saída para um novo dado de teste
% Dados devem estar no formato Nxp
%
% - Entradas:
%       k = numero de vizinhos
%       P = matriz de entrada de treino
%       T1 = matriz de saida de treino
%       Q = matriz de entrada de teste
%       T2 = matriz de saida de teste
% - Saídas
%       y_h = estimação da saida para conjunto de teste
%       EQ2 = Erro de quantização quadrático

[treino,~] = size(P);
[teste,~] = size(Q);
[~,saidas] = size(T2);

y_h = zeros(teste,saidas);
EQ2 = zeros(teste,1);

for i = 1:teste,

    % Calculo das distancias do vetor de teste a cada vetor de treinamento
    vet_dist = zeros(1,treino);
    
    for j = 1:treino,
        x = Q(i,:) - P(j,:);        % vetor diferença       
        vet_dist(j) = norm(x,2);    % norma quadrática
    end
    
    % Ordenamento das distâncias e achar vizinhos mais próximos
    [~,aux1] = sort(vet_dist,2,'ascend');
    Knear = aux1(1:k);

    % Estimação da saída
    y_h(i,:) = mean(T1(Knear,:));
    
    % Cálculo do erro quadrático
    x = y_h(i,:) - T2(i,:);
    EQ2(i) = norm(x,2);
    
end

% End
