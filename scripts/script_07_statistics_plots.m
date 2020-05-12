%% BOXPLOT EXAMPLE

% Geração de um gráfico para análise estatística dos dados

clear;
clc;

%% Gera gráfico de caixa

% linha vermelha: Mediana
% Linhas azuis: 25% a 75%
% Linhas pretas: faixa de valores (sem ser outliers)
% Pontos vermelhos: outliers (plodados individualmente)
% Cada coluna é um boxplot

a = 70; b = 85;                         % Valores mínimos e máximos
Mat_boxplot = a + (b-a).*rand(100,3);   % Geração de dados aleatórios
media = mean(Mat_boxplot);              % Média dos dados

labels = {'MLP' 'ELM' 'MLM'};           % Nomes de cada coluna

figure; boxplot(Mat_boxplot, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
title('Classificaiton Rate')    % Titulo da figura
ylabel('% Accuracy')            % label eixo y
xlabel('Classifiers')           % label eixo x
axis ([0 4 40 100])             % Eixos da figura

grid on                         % "Grade"/"Malha" (melhorar visualização)

hold on
plot(media,'*k')                % Plotar média no mesmo gráfico
hold off

%% Gerar grafico de dispersão

% Geração de um gráfico para análise estatística dos dados
% Compara duas colunas (uma no eixo X, outra no eixo Y)

x1 = (0:0.1:1)';
x2 = (0:0.2:2)';
X = [x1,x2];

scatterplot(X);
xlabel('teste1');
ylabel('teste2');
title('titulo1');

%% Gera barra de erros

% Geração de um gráfico para análise estatística dos dados
% x / y / e - devem ter a mesma dimensao (por isso a funcao "ones")

x = 1:10;
y = sin(x);
e1 = std(y)*ones(size(x));
e2 = std(y)*ones(size(x));
errorbar(x,y,e1,e2)

%% Verificar se dados obedecem determinada distribuição

% QQPLOT

%% END