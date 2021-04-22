%% Get Signals and Time

va = signals2.signals.values(:,1);  % tensao
ia = signals2.signals.values(:,2);  % corrente
cl = signals2.signals.values(:,3);  % carga
w = signals2.signals.values(:,4);   % velocidade
t = signals2.time;                  % vetor de tempo
dt = t(1) - t(2);                   % discretizacao
N = length(w);                      % Number of samples

%% Plot signals

figure;
hold on
plot(t,va,'b')
plot(t,ia,'r')
plot(t,cl,'g')
plot(t,w,'k')
legend('tensao','corrente','carga','velocidade')
hold off

%% Build Regression Matrices

y1 = w(3:N);
y2 = ia(3:N);

A1 = zeros(N-2,6);
A2 = zeros(N-2,6);

for i = 3:N,

    A1(i-2,:) = [w(i-1), w(i-2), ...
                 va(i-1), va(i-2), ...
                 cl(i-1), cl(i-2)];

    A2(i-2,:) = [ia(i-1), ia(i-2), ...
                 va(i-1), va(i-2), ...
                 cl(i-1), cl(i-2)];
end

%% Calculate Regression coeficients

coef1 = pinv(A1)*y1;
coef2 = pinv(A2)*y1;

%% Calcular recursivamente sinais


y_mem1 = zeros(1,2);

for i = 1:N,
    
end

%% Comparar sinais originais com calculados

vel_comp = w - vel2;
MSE = sqrt(sum(vel_comp.^2));

%% END








