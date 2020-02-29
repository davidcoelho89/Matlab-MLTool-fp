%% GRÁFICOS - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

nfig = 0;       % contagem do numero de figuras

%% "ESTILO AMOSTRAGEM"

% stem()          % para plotar ponto com "barras"

%% PLOTAR CURVAS 2D

x = 1:0.1:10;   % gera vetor de 1 a 10, incrementando a cada 0.1
y = x.^2;       % eleva cada ponto do vetor anterior ao quadrado

% indica em qual figura ficará o próximo gráfico
nfig = nfig + 1; figure(nfig)

plot(x,y);      % gera gráfico 2d

%% PLOTAR CURVAS 2D (SEM BORDA CINZA)

x = 0:0.01:1;               % x variando de 0 a 1, de 0.01 em 0.01
y = 2*x;

figure(1)                   % Abre uma janela propria
plot(x,y)                   % faz a curva Y (vertical) por X (horizontal)
set(gcf,'color',[1 1 1])    % Tira o fundo Cinza do Matlab

%% PLOTAR CURVAS 3D

x = 1:0.1:10;   % gera vetor linha [1 x n] de 1 a 10
y = x';         % gera vetor coluna [n x 1] de 1 a 10 (x' = x transposto)

% Gera uma matriz de várias linhas iguais ao vetor linha x, em X2
% Gera uma matriz de várias colunas iguais ao vetor coluna y, em Y2
[X2, Y2] = meshgrid(x,y); % matriz [n x n]

Z2 = X2+Y2;     % faz a soma das duas funções ponto a ponto [n x n]

nfig = nfig + 1;    figure(nfig)

plot3(X2,Y2,Z2);    % Gráfico 3D - pontos e linhas

nfig = nfig + 1;    figure(nfig)

surf(X2,Y2,Z2)      % Gráfico 3D - superfície

nfig = nfig + 1;    figure(nfig)

mesh(X2,Y2,Z2)      % Gráfico 3D - pontos e linhas ligados

nfig = nfig + 1;    figure(nfig)

contour(X2,Y2,Z2)   % Curvas de Contorno (apenas linhas)

nfig = nfig + 1;    figure(nfig)

contourf(X2,Y2,Z2)  % Curvas de Contorno (espaços preenchidos)

%% SUBPLOTS

%help subplot

%% GRAFICOS ONLINE

nfig = nfig + 1;    figure(nfig)

x = 0:pi/100:2*pi;
y = cos(x);
h = plot(x,y,'YdataSource','y');

for k = 1:0.01:5,
   y = cos(x.*k);
   refreshdata(h,'caller')
   drawnow
end

%% PLOTANDO 2 GRAFICOS NA MESMA FIGURA

nfig = nfig + 1;    figure(nfig)

x = 0:pi/20:2*pi;   % Create the x-axis
y1 = 2*sin(x/2);    % Make a vector for 2sin(x/2)
plot(x,y1,'r-')     % Plot 2sin(x/2) as a solid line in red

hold on             % Use hold on to add a second plot to the graph

y2 = cos(2*x);      % Make a vector for cos(2x)
plot(x,y2,'kx')     % Plot cos(2x)

legend({'2sin(0.5x)','cos(2x)'})        % Legend
xlabel('x')                             % X label
ylabel('2sin(0.5x) , cos(2x)')          % Y label
title('This is my second MSUM plot')    % Title

hold off            % Turn off the hold

%% PLOT PROPERTIES 1

% PLOT 1 (COLOR, MARKER, LINESTYLE)

% Main types of markers, line style and colors
% color_array =  {'y','m','c','r','g','b','k','w'}; % rand(1,3) -> can append
% marker_array = {'.','*','o','x','+','s','d','v','^','<','>','p','h'};
% line_style_array = {'-', ':', '-.', '--','(none)'};

% Example of usage
% figure;
% plot(x,y,'Color',color_array{1},'Marker', ...
% marker_array{1},'LineStyle',line_style_array{1})

% PLOT 2 (MARKER AND LINE PROPERTIES)

% % Plot of neurons' grid
% figure;
% for i  = 1:dim(2),
%    plot(C(i,:,2),C(i,:,3),'-ms',...
%    'LineWidth',1,...
%    'MarkerEdgeColor','k',...
%    'MarkerFaceColor','g',...
%    'MarkerSize',5)
% end

%% PLOT PROPERTIES 2

% color_array =  {'y','m','c','r','g','b','k','w'};
% marker_array = {'.','*','o','x','+','s','d','v','^','<','>','p','h'};
% 
% dados = DATA.dados;
% alvos = DATA.alvos;
% 
% figure;
% hold on
% for i = 1:N,
%     plot(dados(1,i),dados(2,i),'Color',color_array{alvos(i)}, ...
%         'LineStyle', marker_array{alvos(i)})
% end
% hold off
% 
% figure;
% hold on
% for i = 1:N,
%     plot(dados(2,i),dados(3,i),'Color',color_array{alvos(i)}, ...
%         'LineStyle', marker_array{alvos(i)})
% end
% hold off
% 
% 
% figure;
% hold on
% for i = 1:N,
%     plot(dados(3,i),dados(4,i),'Color',color_array{alvos(i)}, ...
%         'LineStyle', marker_array{alvos(i)})
% end
% hold off

%% END