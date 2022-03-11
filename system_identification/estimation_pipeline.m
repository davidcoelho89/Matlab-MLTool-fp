%% PIPELINE

clear;
clc;
close;

%% OPTIONS

% ARX(2,1)
lag_y = 2;
lag_u = 1;
p = lag_y + lag_u;

% EXPERIMENT
Nr = 10;    % Numero de repeticoes
NAMES = {'0%','5%','10%','20%','30%'};

% LMS e LMM
Nep = 5;    % Numero de epocas
eta = 0.1;  % Taxa de aprendizado
Kout = 0.3; % Maximo nivel de erro

%% INICIALIZA ACUMULADORES

lms_00_par_acc = zeros(Nr,p);
lms_00_rmse_acc = zeros(Nr,1);
lms_05_par_acc = zeros(Nr,p);
lms_05_rmse_acc = zeros(Nr,1);
lms_10_par_acc = zeros(Nr,p);
lms_10_rmse_acc = zeros(Nr,1);
lms_20_par_acc = zeros(Nr,p);
lms_20_rmse_acc = zeros(Nr,1);
lms_30_par_acc = zeros(Nr,p);
lms_30_rmse_acc = zeros(Nr,1);

lmm_00_par_acc = zeros(Nr,p);
lmm_00_rmse_acc = zeros(Nr,1);
lmm_05_par_acc = zeros(Nr,p);
lmm_05_rmse_acc = zeros(Nr,1);
lmm_10_par_acc = zeros(Nr,p);
lmm_10_rmse_acc = zeros(Nr,1);
lmm_20_par_acc = zeros(Nr,p);
lmm_20_rmse_acc = zeros(Nr,1);
lmm_30_par_acc = zeros(Nr,p);
lmm_30_rmse_acc = zeros(Nr,1);


%% SEQUENCIA

for r = 1:Nr

% Generate Input-Output

N = 500;
n = 1:N;

u_mu = 0;
u_var = 0.5;
u_ts = u_mu + sqrt(u_var)*randn(N,1);

y_coefs = [0.4 -0.6];
u_coefs = 2;
y_ts = arxOutputFromInput(u_ts,y_coefs,u_coefs);

% Generate Time Series with Outliers

y_ts_05 = addTimeSeriesOutilers(y_ts,0.05);
y_ts_10 = addTimeSeriesOutilers(y_ts,0.10);
y_ts_20 = addTimeSeriesOutilers(y_ts,0.20);
y_ts_30 = addTimeSeriesOutilers(y_ts,0.30);

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

% Number of samples used for test

N = length(y_10) - N00_tr;

% LMS Estimation with 0% Outliers

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

lms_00_par_acc(r,:) = ah_lms_00';
lms_00_rmse_acc(r) = sqrt((1/N)*sum(e_lms_00.^2));

% LMS Estimation with 5% Outliers

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

lms_05_par_acc(r,:) = ah_lms_05';
lms_05_rmse_acc(r) = sqrt((1/N)*sum(e_lms_05.^2));

% LMS Estimation with 10% Outliers

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

lms_10_par_acc(r,:) = ah_lms_10';
lms_10_rmse_acc(r) = sqrt((1/N)*sum(e_lms_10.^2));

% LMS Estimation with 20% Outliers

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

lms_20_par_acc(r,:) = ah_lms_20';
lms_20_rmse_acc(r) = sqrt((1/N)*sum(e_lms_20.^2));

% LMS Estimation with 30% Outliers

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

lms_30_par_acc(r,:) = ah_lms_30';
lms_30_rmse_acc(r) = sqrt((1/N)*sum(e_lms_30.^2));

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

lmm_00_par_acc(r,:) = ah_lmm_00';
lmm_00_rmse_acc(r) = sqrt((1/N)*sum(e_lmm_00.^2));

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

lmm_05_par_acc(r,:) = ah_lmm_05';
lmm_05_rmse_acc(r) = sqrt((1/N)*sum(e_lmm_05.^2));

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

lmm_10_par_acc(r,:) = ah_lmm_10';
lmm_10_rmse_acc(r) = sqrt((1/N)*sum(e_lmm_10.^2));

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

lmm_20_par_acc(r,:) = ah_lmm_20';
lmm_20_rmse_acc(r) = sqrt((1/N)*sum(e_lmm_20.^2));

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

lmm_30_par_acc(r,:) = ah_lmm_30';
lmm_30_rmse_acc(r) = sqrt((1/N)*sum(e_lmm_30.^2));

end

%% VERIFY RESULTS

% LMS

lms_00_par_mean = median(lms_00_par_acc);
lms_05_par_mean = median(lms_05_par_acc);
lms_10_par_mean = median(lms_10_par_acc);
lms_20_par_mean = median(lms_20_par_acc);
lms_30_par_mean = median(lms_30_par_acc);

lms_00_rmse_mean = median(lms_00_rmse_acc);
lms_05_rmse_mean = median(lms_05_rmse_acc);
lms_10_rmse_mean = median(lms_10_rmse_acc);
lms_20_rmse_mean = median(lms_20_rmse_acc);
lms_30_rmse_mean = median(lms_30_rmse_acc);

lms_rmse_mean = [lms_00_rmse_mean,lms_05_rmse_mean,lms_10_rmse_mean, ...
                 lms_20_rmse_mean,lms_30_rmse_mean];

% LMM

lmm_00_par_mean = mean(lmm_00_par_acc);
lmm_05_par_mean = mean(lmm_05_par_acc);
lmm_10_par_mean = mean(lmm_10_par_acc);
lmm_20_par_mean = mean(lmm_20_par_acc);
lmm_30_par_mean = mean(lmm_30_par_acc);

lmm_00_rmse_mean = mean(lmm_00_rmse_acc);
lmm_05_rmse_mean = mean(lmm_05_rmse_acc);
lmm_10_rmse_mean = mean(lmm_10_rmse_acc);
lmm_20_rmse_mean = mean(lmm_20_rmse_acc);
lmm_30_rmse_mean = mean(lmm_30_rmse_acc);

lmm_rmse_mean = [lmm_00_rmse_mean,lmm_05_rmse_mean,lmm_10_rmse_mean, ...
                 lmm_20_rmse_mean,lmm_30_rmse_mean];

% Error Boxplot

Mbp_lms = [lms_00_rmse_acc,lms_05_rmse_acc,lms_10_rmse_acc, ...
           lms_20_rmse_acc, lms_30_rmse_acc];

figure; boxplot(Mbp_lms, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Error')
xlabel('Contamination Rate')
title('LMS Error')
axis ([0 6 -0.05 max(max(Mbp_lms))+0.5])

Mbp_lmm = [lmm_00_rmse_acc,lmm_05_rmse_acc,lmm_10_rmse_acc, ...
           lmm_20_rmse_acc, lmm_30_rmse_acc];

figure; boxplot(Mbp_lmm, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Error')
xlabel('Contamination Rate')
title('LMM Error')
axis ([0 6 -0.05 max(max(Mbp_lmm))+0.5])

%% END