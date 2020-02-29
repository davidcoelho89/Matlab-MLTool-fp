%% GRAFICOS DE DISPERSAO - 7 HARM

% David Nascimento Coelho
% Última alteração: 25/01/2015

%% TESTE 1 - Nao normalizados

clear all; close all; clc;

load data_ESPEC_4;          %16 Harmonicas
data = data1';              %dados (1 amostra por linha)
rot = rot';                 %rotulos (1 amostra por linha)

harm = 2*[0.5 1 1.5 2.5 3 5 7];
data = data(:,harm);

% 0.5 e 1.5
X1 = data(1:42,1);
Y1 = data(1:42,3);
X2 = data(43:end,1);
Y2 = data(43:end,3);

figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('0.5fs');
ylabel('1.5fs');
title('Gráfico de Dispersão');

% 2.5 e 7
X1 = data(1:42,4);
Y1 = data(1:42,7);
X2 = data(43:end,4);
Y2 = data(43:end,7);

figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('2.5fs');
ylabel('7fs');
title('Gráfico de Dispersão');

% 3 e 5
X1 = data(1:42,5);
Y1 = data(1:42,6);
X2 = data(43:end,5);
Y2 = data(43:end,6);
figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('3fs');
ylabel('5fs');
title('Gráfico de Dispersão');

%% TESTE 2 - Normalizados por "mu" e "std"

clear all; close all; clc;

load data_ESPEC_4;          %16 Harmonicas
data = data1';              %dados (1 amostra por linha)
data = normalize(data,3);   %normaliza pela média e desvio padrão
rot = rot';                 %rotulos (1 amostra por linha)

harm = 2*[0.5 1 1.5 2.5 3 5 7];
data = data(:,harm);

% 0.5 e 1.5
X1 = data(1:42,1);
Y1 = data(1:42,3);
X2 = data(43:end,1);
Y2 = data(43:end,3);

figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('0.5fs');
ylabel('1.5fs');
title('Gráfico de Dispersão');

% 2.5 e 7
X1 = data(1:42,4);
Y1 = data(1:42,7);
X2 = data(43:end,4);
Y2 = data(43:end,7);

figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('2.5fs');
ylabel('7fs');
title('Gráfico de Dispersão');

% 3 e 5
X1 = data(1:42,5);
Y1 = data(1:42,6);
X2 = data(43:end,5);
Y2 = data(43:end,6);
figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('3fs');
ylabel('5fs');
title('Gráfico de Dispersão');

%% TESTE 3 - Normalizados por "fundamental"

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

% 0.5 e 1.5
X1 = data(1:42,1);
Y1 = data(1:42,2);
X2 = data(43:end,1);
Y2 = data(43:end,2);

figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('0.5fs');
ylabel('1.5fs');
title('Gráfico de Dispersão');

% 2.5 e 7
X1 = data(1:42,3);
Y1 = data(1:42,6);
X2 = data(43:end,3);
Y2 = data(43:end,6);

figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('2.5fs');
ylabel('7fs');
title('Gráfico de Dispersão');

% 3 e 5
X1 = data(1:42,4);
Y1 = data(1:42,5);
X2 = data(43:end,4);
Y2 = data(43:end,5);
figure; plot(X1,Y1,'.b',X2,Y2,'.r');
xlabel('3fs');
ylabel('5fs');
title('Gráfico de Dispersão');

%% END
