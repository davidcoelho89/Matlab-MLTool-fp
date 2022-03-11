%% IDENTIFICACAO DE SISTEMAS

% TC2 - Estimação dos Parâmetros AR
% Autor: David Nascimento Coelho
% Data: 24/02/2022

close;
clear;
clc;

%% (0) Add Folder and Subfolders to Matlab Path

% Get full path of current file
filePath = matlab.desktop.editor.getActiveFilename;

% Use the right symbol for the path                                                                           
if(isunix)   
    symb = '/';                   
else
    symb = '\'; 
end

% Find position of last "\" or "/"
nchar = length(filePath);
for i = 1:nchar
    if(filePath(i) == symb)
        lastSymb = i;
    end
end

folderPath = filePath(1:lastSymb);

% Find position of penultimate "\" or "/"
nchar = length(folderPath);
nchar = nchar-1;
for i = 1:nchar
    if(folderPath(i) == symb)
        lastSymb = i;
    end
end

upperfolderPath = folderPath(1:lastSymb);

% Add folder and subfolders to the path
addpath(genpath(upperfolderPath));

% Clear variables
clear filePath folderPath upperfolderPath i lastSymb nchar symb

%% (1.1) Load Time Series Files and Build Samples

dataSet_name = 'furnas';  % 'emborca'; 'furnas'; 'itumbi';
                        % 'maribond'; 'simao';

dataSet_filename = strcat(dataSet_name,'.dat');

data = load(dataSet_filename);

%% (1.2) Time Series (With Seasonality)

dataSet_title = strcat(dataSet_name,' - Time Series with Seasonality');
histogram_title = strcat(dataSet_name,' - Histogram with Seasonality');

% Transform matrix in time series
[Nyears,Nmonths] = size(data);
data_ts = zeros(1,Nyears*Nmonths);
for i = 1:Nyears
    data_ts((i-1)*Nmonths + 1 : i*Nmonths) = data(i,:);
end

% See Data Series
figure;
plot(data_ts);
title(dataSet_title)
xlabel('Months')
ylabel('Streamflow [m3/s]')
axis([1 Nyears*Nmonths+1 min(data_ts)-1  max(data_ts)+1])

% See Data Series Histogram
figure;
histogram(data_ts)
title(histogram_title)
xlabel('Streamflow [m3/s]')
ylabel('Occurance')

%% (1.3) Time Series (Without Seasonality)

dataSet_title2 = strcat(dataSet_name,' - Time Series without Seasonality');
histogram_title2 = strcat(dataSet_name,' - Histogram without Seasonality');

% Remove seasonality
data_transf = log(data);        % 
data_mean = mean(data_transf);  % 
data_std = std(data_transf);    % 
data2 = zeros(60,12);
for i = 1:60
    data2(i,:) = (data_transf(i,:) - data_mean)./data_std;
end

% Transform matrix in time series
[Nyears,Nmonths] = size(data2);
data_ts2 = zeros(1,Nyears*Nmonths);
for i = 1:Nyears
    data_ts2((i-1)*Nmonths + 1 : i*Nmonths) = data2(i,:);
end

figure;
plot(data_ts2);
title(dataSet_title2)
xlabel('Months')
ylabel('Streamflow [m3/s]')
axis([1 Nyears*Nmonths+1 min(data_ts2)-1  max(data_ts2)+1])

figure;
histogram(data_ts2)
title(histogram_title2)
xlabel('Streamflow [m3/s]')
ylabel('Occurance')

%% (1.4) Graphics

figure;

subplot(2,2,1);
plot(data_ts);
title(dataSet_title)
xlabel('Months')
ylabel('Streamflow [m3/s]')
axis([1 Nyears*Nmonths+1 min(data_ts)-1  max(data_ts)+1])

subplot(2,2,2);
histogram(data_ts)
title(histogram_title)
xlabel('Streamflow [m3/s]')
ylabel('Occurance')

subplot(2,2,3);
plot(data_ts2);
title(dataSet_title2)
xlabel('Months')
ylabel('Streamflow [m3/s]')
axis([1 Nyears*Nmonths+1 min(data_ts2)-1  max(data_ts2)+1])

subplot(2,2,4);
histogram(data_ts2)
title(histogram_title2)
xlabel('Streamflow [m3/s]')
ylabel('Occurance')

%% (2) ACF and PACF

TAUmax = 24;                % Maximo lag da FAC

% TS Without seazonality

fac = acvcf(data_ts2,TAUmax);

figure;
stem(fac);
title('Autocorrelation Function')
xlabel('Lag')
ylabel('Autocorrelation')
axis([-1, TAUmax+1, min(fac)-0.1, max(fac)+0.1])

facp = pacf(data_ts2,TAUmax);

N = length(facp);           % 
x = 0:N-1;                  % 
rel = 0.1;                  % 
line_rel1 = rel*ones(1,N);  % 
line_rel2 = -rel*ones(1,N); % 

aux_vector_min = [-rel,facp'];
aux_vector_max = [rel,facp'];

figure;
stem(facp);
hold on
plot(x,line_rel1,'r.')
plot(x,line_rel2,'r.')
hold off
title('Partial Autocorrelation Function')
xlabel('Lag')
ylabel('Partial Autocorrelation')
axis([-1, N+1, min(aux_vector_min)-0.1, max(aux_vector_max)+0.1])

%% (3) MODELO AR: DETERMINAÇÃO DA ORDEM

p_max = 20;
p_vet = 1:p_max;
evar_vet = zeros(p_max,1);
Ne_vet = zeros(p_max,1);
aic_vet = zeros(p_max,1);
bic_vet = zeros(p_max,1);
fpe_vet = zeros(p_max,1);
mdl_vet = zeros(p_max,1);

for p = 1:p_max
    [X,y] = regressionMatrixFromTS(data_ts2,p);
    ah_MQ = (X'*X)\(X'*y);
    yh = X * ah_MQ;
    e = y - yh;
    evar_vet(p) = var(e);
    Ne_vet(p) = length(e);

%     aic_vet(p) = log(evar_vet(p)) + (2*p)/Ne_vet(p);
%     bic_vet(p) = log(evar_vet(p)) + p*log(Ne_vet(p))/Ne_vet(p);
%     fpe_vet(p) = log(evar_vet(p)) + log((Ne_vet(p)+p)/(Ne_vet(p)-p));
%     mdl_vet(p) = log(evar_vet(p)) + p*log(Ne_vet(p))/(2*Ne_vet(p));
    
    aic_vet(p) = Ne_vet(p)*log(evar_vet(p)) + (2*p);
    bic_vet(p) = Ne_vet(p)*log(evar_vet(p)) + p*log(Ne_vet(p));
    fpe_vet(p) = Ne_vet(p)*(log(evar_vet(p)) + log((Ne_vet(p)+p)/(Ne_vet(p)-p)));
    mdl_vet(p) = Ne_vet(p)*log(evar_vet(p)) + p*log(Ne_vet(p))/2;
end

[~,min_aic] = min(aic_vet);
[~,min_bic] = min(bic_vet);
[~,min_fpe] = min(fpe_vet);
[~,min_mdl] = min(mdl_vet);

figure;
ap = plot(p_vet,aic_vet,'k.-');
hold on
bp = plot(p_vet,bic_vet,'r-');
fp = plot(p_vet,fpe_vet,'b-');
mp = plot(p_vet,mdl_vet,'y-');
hold off
title('Validation Criteria')
xlabel('Order (p)')
ylabel('Value')
legend({'AIC','BIC','FPE','MDL'},'Location','northwest')

%% (4) MODELO AR: ESTIMAÇÃO DOS PARAMETROS

% Ordem definida na questão anterior.
p = 2;

% Estimação por MQ
[X,y] = regressionMatrixFromTS(data_ts2,p);
ah_MQ = (X'*X)\(X'*y);

% Estimação por Yule-Walker
ah_aryule = aryule(data_ts2,p);
ah_aryule = -ah_aryule(2:end)';

%% (5.1) ERRO: HISTOGRAMA E RESIDUOS

TAUmax = 50;

yh_MQ = X * ah_MQ;
yh_aryule = X * ah_aryule;

e_MQ = y - yh_MQ;
e_aryule = y - yh_aryule;

figure;
histogram(e_MQ);
title('Histograma dos residuos')
xlabel('Values')
ylabel('Mininos Quadrados')

figure;
histogram(e_aryule);
title('Histograma dos residuos')
xlabel('Values')
ylabel('Yule-Walker')

fac_MQ = acvcf(e_MQ,TAUmax);
figure;
stem(fac_MQ);
title('Residues Autocorrelation Function')
xlabel('Lag')
ylabel('Mininos Quadrados')

fac_YW = acvcf(e_aryule,TAUmax);
figure;
stem(fac_YW);
title('Residues Autocorrelation Function')
xlabel('Lag')
ylabel('Yule-Walker')

%% (5.2) ERRO: HISTOGRAMA E RESIDUOS

figure;

subplot(1,2,1);
histogram(e_MQ);
title('Histograma dos residuos')
xlabel('Valores')
ylabel('Mininos Quadrados')

subplot(1,2,2);
stem(fac_MQ);
title('Resíduos - Função de Autocorrelação')
xlabel('Lag')
ylabel('Mininos Quadrados')

figure; 

subplot(1,2,1);
histogram(e_aryule);
title('Histograma dos residuos')
xlabel('Valores')
ylabel('Yule-Walker')

subplot(1,2,2);
stem(fac_YW);
title('Resíduos - Função de Autocorrelação')
xlabel('Lag')
ylabel('Yule-Walker')

%% END