%% BOXPLOT EXAMPLE

% Geração de um gráfico para análise estatística dos dados
% Última Alteração: 02/01/2014

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

%% END