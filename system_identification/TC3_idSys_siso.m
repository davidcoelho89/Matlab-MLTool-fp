%% IDENTIFICACAO DE SISTEMAS

% TC3 - Identificação de um sistema ARX (SISO)
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

%% (1.1) INPUT-OUTPUT SYSTEM (ARX) - INPUT: PRBS

% SAMPLES VECTOR

N = 500;
n = 1:N;

min_n = min(n);
max_n = max(n);

% INPUT: PRBS

u_ts = round(rand(N,1));

min_u = min(u_ts);
max_u = max(u_ts);

% Plot Input

figure;
plot(n,u_ts,'r-')
title('Input: PRBS')
xlabel('Sample')
ylabel('Amplitude')
axis([min_n-1, max_n+1, min_u-0.1, max_u+0.1])

% OUTPUT: y(k) = 0.4y(k-1) – 0.6y(k-2) + 2u(k-1) + v(k)

y_coefs = [0.4 -0.6];
u_coefs = 2;
noise_var = 0.01;

y_ts = arxOutputFromInput(u_ts,y_coefs,u_coefs,noise_var);

min_y = min(y_ts);
max_y = max(y_ts);

% Plot Output
figure;
plot(n,y_ts,'r-')
title('y(k) = 0,4y(k-1) – 0,6y(k-2) + 2u(k-1)')
xlabel('Sample')
ylabel('Amplitude')
axis([min_n-1, max_n+1, min_y-0.1, max_y+0.1])

%% (1.2) PARAMETER ESTIMATION USING OLS

% OLS Hyperparameters

lambda = 0.0001;

% Vector of Validation Indexes

aic_vet = zeros(3,1);
bic_vet = zeros(3,1);

% ARX(1,1)

lag_y = 1;
lag_u = 1;
p = lag_y + lag_u;

[X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u);

Ntr = ceil(length(y)/2);
Xtr = X(1:Ntr,:);
ytr = y(1:Ntr,:);
Xts = X(Ntr+1:end,:);
yts = y(Ntr+1:end,:);

ah_MQ_11 = (Xtr'*Xtr + lambda*eye(p))\(Xtr'*ytr);

yh = Xts * ah_MQ_11;
e = yts - yh;
evar = var(e);
Ne = length(e);
aic_vet(1) = log(evar) + (2*p)/Ne;
bic_vet(1) = log(evar) + p*log(Ne)/Ne;

% ARX(2,1)

lag_y = 2;
lag_u = 1;
p = lag_y + lag_u;

[X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u);

Ntr = ceil(length(y)/2);
Xtr = X(1:Ntr,:);
ytr = y(1:Ntr,:);
Xts = X(Ntr+1:end,:);
yts = y(Ntr+1:end,:);

ah_MQ_21 = (Xtr'*Xtr + lambda*eye(p))\(Xtr'*ytr);

yh = Xts * ah_MQ_21;
e = yts - yh;
evar = var(e);
Ne = length(e);
aic_vet(2) = log(evar) + (2*p)/Ne;
bic_vet(2) = log(evar) + p*log(Ne)/Ne;

% ARX(2,2)

lag_y = 2;
lag_u = 2;
p = lag_y + lag_u;

[X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u);

Ntr = ceil(length(y)/2);
Xtr = X(1:Ntr,:);
ytr = y(1:Ntr,:);
Xts = X(Ntr+1:end,:);
yts = y(Ntr+1:end,:);

ah_MQ_22 = (Xtr'*Xtr + lambda*eye(p))\(Xtr'*ytr);

yh = Xts * ah_MQ_22;
e = yts - yh;
evar = var(e);
Ne = length(e);
aic_vet(3) = log(evar) + (2*p)/Ne;
bic_vet(3) = log(evar) + p*log(Ne)/Ne;

%% (1.3) RESIDUE ANALYSIS (Histogram and FAC) - OLS

% ARX(2,1)

lag_y = 2;
lag_u = 1;
% p = lag_y + lag_u;

[X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u);

Ntr = ceil(length(y)/2);
Nts = length(y) - Ntr;
Xtr = X(1:Ntr,:);
ytr = y(1:Ntr,:);
Xts = X(Ntr+1:end,:);
yts = y(Ntr+1:end,:);

% Generate Outputs from training and test data

yh_MQ = Xtr * ah_MQ_21;
e_MQ = ytr - yh_MQ;

yh_MQ_ts = Xts * ah_MQ_21;
e_MQ_ts = yts - yh_MQ_ts;

% RMSE

RMSE_tr = sqrt((1/Ntr)*sum(e_MQ.^2));
RMSE_ts = sqrt((1/Nts)*sum(e_MQ_ts.^2));

% Generate Histogram

figure;
histfit(e_MQ);
title('Histograma dos residuos - OLS')
xlabel('Values')

% Generate Aucorrelation Function

TAUmax = 50;
fac = acvcf(e_MQ,TAUmax);

figure;
stem(fac);
title('Residues Autocorrelation Function - OLS')
axis([-1,TAUmax+3,min(fac)-0.1,max(fac)+0.1])
xlabel('Lag')

%% (1.4) PARAMETER ESTIMATION USING LMS

% ARX(2,1)

lag_y = 2;
lag_u = 1;
p = lag_y + lag_u;

[X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u);
N = length(y);

Ntr = ceil(N/2);
Nts = N - Ntr;
Xtr = X(1:Ntr,:);
ytr = y(1:Ntr,:);
Xts = X(Ntr+1:end,:);
yts = y(Ntr+1:end,:);

% LMS

Nep = 1;
eta = 0.1;
ah_lms = randn(p,1);

ah_lms_acc = zeros(p,Nep*N+1);
ah_lms_acc(:,1) = ah_lms;
cont = 1;

for ep = 1:Nep

    for n = 1:Ntr
        xn = Xtr(n,:)';
        yh = ah_lms' * xn;
        er = ytr(n) - yh;
        ah_lms = ah_lms + eta * er * xn / (xn'*xn);
        cont = cont+1;
        ah_lms_acc(:,cont) = ah_lms;
    end

end

% Plot Parameters adaptation

figure;
plot(ah_lms_acc(1,:),'r-');
hold on
plot(ah_lms_acc(2,:),'b-');
plot(ah_lms_acc(3,:),'k-');
hold off
title('Parameters Adaptation: y(k) = a1*y(k-1) + a2*y(k-2) + b1*u(k-1)')
axis([-1,Nep*Ntr,-1,2.5])
legend({'a1','a2','b1'},'Location','northwest')

% Calculate Residues

yh_lms = Xtr * ah_lms;
e_lms = ytr - yh_lms;
RMSE_lms_tr = sqrt((1/Ntr)*sum(e_lms.^2));

% Plot Residues Histogram

figure;
histfit(e_lms);
title('Histograma dos residuos - LMS')
xlabel('Values')

% Plot Residues Autocorrelation Function

TAUmax = 50;
fac = acvcf(e_lms,TAUmax);

figure;
stem(fac);
title('Residues Autocorrelation Function')
axis([-1,TAUmax+3,min(fac)-0.1,max(fac)+0.1])
xlabel('Lag')

% Calculate Prevision errors

yh_lms_ts = Xts * ah_lms;
e_lms_ts = yts - yh_lms_ts;
RMSE_lms_ts = sqrt((1/Nts)*sum(e_lms_ts.^2));

%% PARAMETER ESTIMATION USING RLS



%% INPUT-OUTPUT SYSTEM (ARX) - INPUT AWGN

% SAMPLES VECTOR

N = 500;
n = 1:N;

% INPUT: AWGN

u_mu = 0;
u_var = 0.5;
u_ts = u_mu + sqrt(u_var)*randn(N,1);

% Plot Input

figure;
plot(n,u_ts,'r-')
title('Input: AWGN. u(k) ~ N(0,0.5)')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(u_ts)-0.1,max(u_ts)+0.1])

% Plot Input Autocorrelation Function

TAUmax = 50;
fac = acvcf(u_ts,TAUmax);

figure;
stem(fac);
title('Autocorrelation Function')

% OUTPUT: y(k) = 0.4y(k-1) – 0.6y(k-2) + 2u(k-1) + v(k)

y_coefs = [0.4 -0.6];
u_coefs = 2;
y_ts = arxOutputFromInput(u_ts,y_coefs,u_coefs);

% Plot Output

figure;
plot(n,y_ts,'r-')
title('y(k) = 0,4y(k-1) – 0,6y(k-2) + 2u(k-1)')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(y_ts)-0.1,max(y_ts)+0.1])

%% ADD OUTLIERS TO OUTPUT

% Generate Time Series

y_ts_05 = addTimeSeriesOutilers(y_ts,0.05);
y_ts_10 = addTimeSeriesOutilers(y_ts,0.10);
y_ts_20 = addTimeSeriesOutilers(y_ts,0.20);
y_ts_30 = addTimeSeriesOutilers(y_ts,0.30);

% Generate Graphic

figure;

subplot(2,2,1);
plot(n,y_ts_05,'r-')
title('y(k) whith 5% of noise')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(y_ts_05)-0.1,max(y_ts_05)+0.1])

subplot(2,2,2);
plot(n,y_ts_10,'r-')
title('y(k) whith 10% of noise')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(y_ts_10)-0.1,max(y_ts_10)+0.1])

subplot(2,2,3);
plot(n,y_ts_20,'r-')
title('y(k) whith 20% of noise')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(y_ts_20)-0.1,max(y_ts_20)+0.1])

subplot(2,2,4);
plot(n,y_ts_30,'r-')
title('y(k) whith 30% of noise')
xlabel('Sample')
ylabel('Amplitude')
axis([min(n)-1,max(n)+1,min(y_ts_30)-0.1,max(y_ts_30)+0.1])

%% GENERATE REGRESSION MATRICES (Training and Test)

% ARX(2,1)

lag_y = 2;
lag_u = 1;
p = lag_y + lag_u;

% Regression Matrix from Noiseless Signal

[X_00,y_00] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u);
N00_tr = ceil(length(y_00)/2);
X_00_tr = X_00(1:N00_tr,:);
X_00_ts = X_00(N00_tr+1:end,:);
y_00_tr = y_00(1:N00_tr,:);
y_00_ts = y_00(N00_tr+1:end,:);

% Regression Matrix from Noisy Signal (5%)

[X_05,y_05] = regressionMatrixFromSISO(y_ts_05,u_ts,lag_y,lag_u);
N05_tr = ceil(length(y_05)/2);
X_05_tr = X_05(1:N05_tr,:);
X_05_ts = X_05(N05_tr+1:end,:);
y_05_tr = y_05(1:N05_tr,:);
y_05_ts = y_05(N05_tr+1:end,:);

% Regression Matrix from Noisy Signal (10%)

[X_10,y_10] = regressionMatrixFromSISO(y_ts_10,u_ts,lag_y,lag_u);
N10_tr = ceil(length(y_10)/2);
X_10_tr = X_10(1:N10_tr,:);
X_10_ts = X_10(N10_tr+1:end,:);
y_10_tr = y_10(1:N10_tr,:);
y_10_ts = y_10(N10_tr+1:end,:);

% Regression Matrix from Noisy Signal (20%)

[X_20,y_20] = regressionMatrixFromSISO(y_ts_20,u_ts,lag_y,lag_u);
N20_tr = ceil(length(y_20)/2);
X_20_tr = X_20(1:N20_tr,:);
X_20_ts = X_20(N20_tr+1:end,:);
y_20_tr = y_20(1:N20_tr,:);
y_20_ts = y_20(N20_tr+1:end,:);

% Regression Matrix from Noisy Signal (30%)

[X_30,y_30] = regressionMatrixFromSISO(y_ts_30,u_ts,lag_y,lag_u);
N30_tr = ceil(length(y_30)/2);
X_30_tr = X_30(1:N30_tr,:);
X_30_ts = X_30(N30_tr+1:end,:);
y_30_tr = y_30(1:N30_tr,:);
y_30_ts = y_30(N30_tr+1:end,:);

%% PARAMETER ESTIMATION USING LMS - WITH OUTLIERS

% Hyperparameters

Nep = 5;
eta = 0.1;

% Estimation with 0% Outliers

ah_lms_00 = randn(p,1);
for ep = 1:Nep

    for n = 1:N00_tr
        xn = X_00_tr(n,:)';
        yh = ah_lms_00' * xn;
        er = y_00_tr(n) - yh;
        ah_lms_00 = ah_lms_00 + eta * er * xn / (xn'*xn);
    end

end

yh_lms_00 = X_00_ts * ah_lms_00;
e_lms_00 = y_00_ts - yh_lms_00;

lms_rmse_00 = sqrt(sum(e_lms_00.^2));

figure;
plot(y_00_ts,'b.-');
hold on
plot(yh_lms_00,'r-');
hold off

% Estimation with 5% Outliers

ah_lms_05 = randn(p,1);
for ep = 1:Nep

    for n = 1:N05_tr
        xn = X_05_tr(n,:)';
        yh = ah_lms_05' * xn;
        er = y_05_tr(n) - yh;
        ah_lms_05 = ah_lms_05 + eta * er * xn / (xn'*xn);
    end

end

yh_lms_05 = X_05_ts * ah_lms_05;
e_lms_05 = y_05_ts - yh_lms_05;

lms_rmse_05 = sqrt(sum(e_lms_05.^2));

figure;
plot(y_05_ts,'b.-');
hold on
plot(yh_lms_05,'r-');
hold off

% Estimation with 10% Outliers

ah_lms_10 = randn(p,1);
for ep = 1:Nep

    for n = 1:N10_tr
        xn = X_10_tr(n,:)';
        yh = ah_lms_10' * xn;
        er = y_10_tr(n) - yh;
        ah_lms_10 = ah_lms_10 + eta * er * xn / (xn'*xn);
    end

end

yh_lms_10 = X_10_ts * ah_lms_10;
e_lms_10 = y_10_ts - yh_lms_10;

lms_rmse_10 = sqrt(sum(e_lms_10.^2));

figure;
plot(y_10_ts,'b.-');
hold on
plot(yh_lms_10,'r-');
hold off

% Estimation with 20% Outliers

ah_lms_20 = randn(p,1);
for ep = 1:Nep

    for n = 1:N20_tr
        xn = X_20_tr(n,:)';
        yh = ah_lms_20' * xn;
        er = y_20_tr(n) - yh;
        ah_lms_20 = ah_lms_20 + eta * er * xn / (xn'*xn);
    end

end

yh_lms_20 = X_20_ts * ah_lms_20;
e_lms_20 = y_20_ts - yh_lms_20;

lms_rmse_20 = sqrt(sum(e_lms_20.^2));

figure;
plot(y_20_ts,'b.-');
hold on
plot(yh_lms_20,'r-');
hold off

% Estimation with 30% Outliers

ah_lms_30 = randn(p,1);
for ep = 1:Nep

    for n = 1:N30_tr
        xn = X_30_tr(n,:)';
        yh = ah_lms_30' * xn;
        er = y_30_tr(n) - yh;
        ah_lms_30 = ah_lms_30 + eta * er * xn / (xn'*xn);
    end

end

yh_lms_30 = X_30_ts * ah_lms_30;
e_lms_30 = y_30_ts - yh_lms_30;

lms_rmse_30 = sqrt(sum(e_lms_30.^2));

figure;
plot(y_30_ts,'b.-');
hold on
plot(yh_lms_30,'r-');
hold off

%% PARAMETER ESTIMATION USING RLS - WITH OUTLIERS



%% PARAMETER ESTIMATION USING LMM - WITH OUTLIERS

% Hyperparameters

Nep = 5;
eta = 0.1;
Kout = 0.3; 

% Estimation with 0% Outliers

ah_lmm_00 = randn(p,1);
for ep = 1:Nep

    for n = 1:N00_tr
        xn = X_00_tr(n,:)';
        yh = ah_lmm_00' * xn;
        er = y_00_tr(n) - yh;
        
        if(abs(er) < Kout)
            q_lmm = 1;
        else
            q_lmm = Kout/abs(er);
        end
        
        ah_lmm_00 = ah_lmm_00 + eta * q_lmm * er * xn / (xn'*xn);
    end

end

yh_lmm_00 = X_00_ts * ah_lmm_00;
e_lmm_00 = y_00_ts - yh_lmm_00;

lmm_rmse_00 = sqrt(sum(e_lmm_00.^2));

figure;
plot(y_00_ts,'b.-');
hold on
plot(yh_lmm_00,'r-');
hold off

% Estimation with 5% Outliers

ah_lmm_05 = randn(p,1);
for ep = 1:Nep

    for n = 1:N05_tr
        xn = X_05_tr(n,:)';
        yh = ah_lmm_05' * xn;
        er = y_05_tr(n) - yh;
        
        if(abs(er) < Kout)
            q_lmm = 1;
        else
            q_lmm = Kout/abs(er);
        end
        
        ah_lmm_05 = ah_lmm_05 + eta * q_lmm * er * xn / (xn'*xn);
    end

end

yh_lmm_05 = X_05_ts * ah_lmm_05;
e_lmm_05 = y_05_ts - yh_lmm_05;

lmm_rmse_05 = sqrt(sum(e_lmm_05.^2));

figure;
plot(y_05_ts,'b.-');
hold on
plot(yh_lmm_05,'r-');
hold off

% Estimation with 10% Outliers

ah_lmm_10 = randn(p,1);
for ep = 1:Nep

    for n = 1:N10_tr
        xn = X_10_tr(n,:)';
        yh = ah_lmm_10' * xn;
        er = y_10_tr(n) - yh;
        
        if(abs(er) < Kout)
            q_lmm = 1;
        else
            q_lmm = Kout/abs(er);
        end
        
        ah_lmm_10 = ah_lmm_10 + eta * q_lmm * er * xn / (xn'*xn);
    end

end

yh_lmm_10 = X_10_ts * ah_lmm_10;
e_lmm_10 = y_10_ts - yh_lmm_10;

lmm_rmse_10 = sqrt(sum(e_lmm_10.^2));

figure;
plot(y_10_ts,'b.-');
hold on
plot(yh_lmm_10,'r-');
hold off

% Estimation with 20% Outliers

ah_lmm_20 = randn(p,1);
for ep = 1:Nep

    for n = 1:N20_tr
        xn = X_20_tr(n,:)';
        yh = ah_lmm_20' * xn;
        er = y_20_tr(n) - yh;
        
        if(abs(er) < Kout)
            q_lmm = 1;
        else
            q_lmm = Kout/abs(er);
        end
        
        ah_lmm_20 = ah_lmm_20 + eta * q_lmm * er * xn / (xn'*xn);
    end

end

yh_lmm_20 = X_20_ts * ah_lmm_20;
e_lmm_20 = y_20_ts - yh_lmm_20;

lmm_rmse_20 = sqrt(sum(e_lmm_20.^2));

figure;
plot(y_20_ts,'b.-');
hold on
plot(yh_lmm_20,'r-');
hold off

% Estimation with 30% Outliers

ah_lmm_30 = randn(p,1);
for ep = 1:Nep

    for n = 1:N30_tr
        xn = X_30_tr(n,:)';
        yh = ah_lmm_30' * xn;
        er = y_30_tr(n) - yh;
        
        if(abs(er) < Kout)
            q_lmm = 1;
        else
            q_lmm = Kout/abs(er);
        end
        
        ah_lmm_30 = ah_lmm_30 + eta * q_lmm * er * xn / (xn'*xn);
    end

end

yh_lmm_30 = X_30_ts * ah_lmm_30;
e_lmm_30 = y_30_ts - yh_lmm_30;

lmm_rmse_30 = sqrt(sum(e_lmm_30.^2));

figure;
plot(y_30_ts,'b.-');
hold on
plot(yh_lmm_30,'r-');
hold off

%% PARAMETER ESTIMATION USING RLM - WITH OUTLIERS



%% END