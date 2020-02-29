%% ANALISE ESTATISTICA DOS DADOS - 7 HARM

% David Nascimento Coelho
% Última alteração: 25/01/2015

%% TESTE 01 - Boxplot dos dados (não normalizados)

clear all; close all; clc;

load data_ESPEC_4;          %16 Harmonicas
data = data1';              %dados (1 amostra por linha)
rot = rot';                 %rotulos (1 amostra por linha)

harm = 2*[0.5 1 1.5 2.5 3 5 7];
data = data(:,harm);

% Com fundamental - dados misturados

% Estatisticas dos dados
Cx1 = cov(data);              % matriz de covariancia
cond_Cx1 = cond(Cx1);         % condicionamento
rcond_Cx1 = rcond(Cx1);       % condicionamento (0 a 1)
det_Cx1 = det(Cx1);           % determinante
inv_Cx1 = inv(Cx1);           % inversa

labels = {'0.5' '1' '1.5' '2.5' '3' '5' '7'};

figure; boxplot(data, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de todos os Dados') % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
% axis ([0 4 40 100])               % Eixos X e Y da figura

% Com fundamental - dados separados (falha x nao falha)
data_normal = data(1:42,:);
data_falha = data(43:end,:);

% Estatisticas dados Normais
Cx2 = cov(data_normal);       % matriz de covariancia
cond_Cx2 = cond(Cx2);         % condicionamento
rcond_Cx2 = rcond(Cx2);       % condicionamento (0 a 1)
det_Cx2 = det(Cx2);           % determinante
inv_Cx2 = inv(Cx2);           % inversa

% Estatisticas dados Falhas
Cx3 = cov(data);              % matriz de covariancia
cond_Cx3 = cond(Cx3);         % condicionamento
rcond_Cx3 = rcond(Cx3);       % condicionamento (0 a 1)
det_Cx3 = det(Cx3);           % determinante
inv_Cx3 = inv(Cx3);           % inversa

figure; boxplot(data_normal, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados Normais')  % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 17 -0.1 1.2])              % Eixos X e Y da figura

figure; boxplot(data_falha, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados de Falhas')% Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 17 -0.1 1.2])              % Eixos X e Y da figura

% Sem fundamental - dados misturados
harm = 2*[0.5 1.5 2.5 3 5 7];
data = data(:,harm);

labels = {'0.5' '1.5' '2.5' '3' '5' '7'};

figure; boxplot(data, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Distribuição dos Atributos') % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
% axis ([0 4 40 100])               % Eixos X e Y da figura

% Sem fundamental - dados separados (falha x nao-falha)
data_normal = data(1:42,:);
data_falha = data(43:end,:);

figure; boxplot(data_normal, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados Normais')  % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 16 -0.01 0.05])          % Eixos X e Y da figura

figure; boxplot(data_falha, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados de Falhas')% Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 16 -0.01 0.05])          % Eixos X e Y da figura

%% TESTE 02 - Boxplot dos dados (normalizados por "mu" e "std")

clear all; close all; clc;

load data_ESPEC_4;          %16 Harmonicas
data = data1';              %dados (1 amostra por linha)
data = normalize(data,3);   %normaliza pela média e desvio padrão
rot = rot';                 %rotulos (1 amostra por linha)

harm = 2*[0.5 1 1.5 2.5 3 5 7];
data = data(:,harm);

% Estatisticas dos dados
Cx1 = cov(data);              % matriz de covariancia
cond_Cx1 = cond(Cx1);         % condicionamento
rcond_Cx1 = rcond(Cx1);       % condicionamento (0 a 1)
det_Cx1 = det(Cx1);           % determinante
inv_Cx1 = inv(Cx1);           % inversa

labels = {'0.5' '1' '1.5' '2.5' '3' '5' '7'};

figure; boxplot(data, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de todos os Dados') % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 17 -2.5 6.5])              % Eixos X e Y da figura

% Com fundamental - dados separados (falha x nao falha)
data_normal = data(1:42,:);
data_falha = data(43:end,:);

% Estatisticas dados Normais
Cx2 = cov(data_normal);       % matriz de covariancia
cond_Cx2 = cond(Cx2);         % condicionamento
rcond_Cx2 = rcond(Cx2);       % condicionamento (0 a 1)
det_Cx2 = det(Cx2);           % determinante
inv_Cx2 = inv(Cx2);           % inversa

% Estatisticas dados Falhas
Cx3 = cov(data);              % matriz de covariancia
cond_Cx3 = cond(Cx3);         % condicionamento
rcond_Cx3 = rcond(Cx3);       % condicionamento (0 a 1)
det_Cx3 = det(Cx3);           % determinante
inv_Cx3 = inv(Cx3);           % inversa

figure; boxplot(data_normal, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados Normais')  % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 17 -2.5 6.5])              % Eixos X e Y da figura

figure; boxplot(data_falha, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados de Falhas')% Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 17 -2.5 6.5])              % Eixos X e Y da figura

%% TESTE 03 - Boxplot dos dados (normalizados pela fundamental)

clear all; close all; clc;

load data_ESPEC_4;          %16 Harmonicas
data = data1';              %dados (1 amostra por linha)
[linD, colD] = size(data);  %numero de linhas e colunas da matriz
data = normalize(data,3);   %normaliza pela média e desvio padrão
rot = rot';                 %rotulos (1 amostra por linha)

harm = 2*[0.5 1.5 2.5 3 5 7];
fundamental = 2*[1];

fs = data(:,fundamental);
data = data(:,harm);

% normaliza pela fundamental
for i = 1:linD,
   data(i,:) = data(i,:)/fs(i);
end

% Estatisticas dos dados
Cx1 = cov(data);              % matriz de covariancia
cond_Cx1 = cond(Cx1);         % condicionamento
rcond_Cx1 = rcond(Cx1);       % condicionamento (0 a 1)
det_Cx1 = det(Cx1);           % determinante
inv_Cx1 = inv(Cx1);           % inversa

labels = {'0.5' '1.5' '2.5' '3' '5' '7'};

figure; boxplot(data, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de todos os Dados') % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 16 -80 65])                % Eixos X e Y da figura

% Com fundamental - dados separados (falha x nao falha)
data_normal = data(1:42,:);
data_falha = data(43:end,:);

% Estatisticas dados Normais
Cx2 = cov(data_normal);       % matriz de covariancia
cond_Cx2 = cond(Cx2);         % condicionamento
rcond_Cx2 = rcond(Cx2);       % condicionamento (0 a 1)
det_Cx2 = det(Cx2);           % determinante
inv_Cx2 = inv(Cx2);           % inversa

% Estatisticas dados Falhas
Cx3 = cov(data);              % matriz de covariancia
cond_Cx3 = cond(Cx3);         % condicionamento
rcond_Cx3 = rcond(Cx3);       % condicionamento (0 a 1)
det_Cx3 = det(Cx3);           % determinante
inv_Cx3 = inv(Cx3);           % inversa

figure; boxplot(data_normal, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados Normais')  % Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 16 -80 65])                % Eixos X e Y da figura

figure; boxplot(data_falha, 'label', labels);
set(gcf,'color',[1 1 1])            % Tira o fundo Cinza do Matlab
title('Variação de dados de Falhas')% Titulo da figura
ylabel('Valor')                     % label eixo y
xlabel('Atributos')                 % label eixo x
axis ([0 16 -80 65])                % Eixos X e Y da figura

%% END
