%% DC MOTOR TRANSFER FUNCTION ESTIMATION

clear;
clc;

%% Generate continuous sin and cos

t = 0:0.01:10;
w = 0.7*pi;

signal_in = cos(w*t);
signal_out = sin(w*t);
figure; plot(t,signal_in);
figure; plot(t,signal_out);

%% Generate random Signals

% For reproducibility
rng('default')

% White Gaussian Noise
N = 100;
mu = 1;
sigma = 2;
r = mu + sigma.*randn(N,1);
figure; plot(r);

% Add White Gaussian Noise
t = (0:0.1:10)';
u = sawtooth(t);
snr = 10;
y = awgn(u,snr,'measured');
figure; plot(t,[u y]);
legend('Original Signal','Signal with AWGN')

%% Generate system input and output

% x(n) = sinal aleatorio!

tmax = 10;  	% tempo total: 10s
dt = 0.001;     % taxa de amostragem: 1000 Hz
t = 0:dt:tmax;	% vetor de tempo contínuo
N = length(t);  % quantidade de amostras
n = 1:N;        % vetor de tempo discreto

rng('default')
mu = 1;
sigma = 2;
u = mu + sigma.*randn(1,N);

figure; stem(n,u,'.');

% Equacao do sistema!
% y(n) = (3/4)*y(n-1) + (-1/8)*y(n-2) + (1/4)*u(n-1) + (1/4)*u(n-2);

input_max_delay = 2;
output_max_delay = 2;
Ns = length(u);

Ud = zeros(input_max_delay,1);
Yd = zeros(output_max_delay,1);

A = [3/4 -1/8];
B = [1/4 1/4];

y = zeros(1,N);

for i = 1:N
    % Build output
    y(i) = A*Yd + B*Ud;
    % Uptade memory
    Ud = [u(i);Ud(1:end-1)];
    Yd = [y(i);Yd(1:end-1)];
end 

figure; stem(n,y,'.');

%% Generate prediction vectors

% X*v = e

X = zeros(N-output_max_delay,output_max_delay+input_max_delay);
e = zeros(N-output_max_delay,1);

for i = 1:(N-output_max_delay)
    e(i) = y(i+output_max_delay);
    X(i,:) = [ flip( y( i : (i+output_max_delay-1)) ) , ...
               flip( u( i : (i+input_max_delay-1) ) ) ];
end

%% Estimate Coefficients

v = ((X'*X)^(-1))*X'*e;

%% Predict 

input_max_delay = 2;
output_max_delay = 2;
N = length(u);

% Vectors with latest input and output samples

Ud = zeros(input_max_delay,1);
Yd = zeros(output_max_delay,1);

% Vectors with coefficients

A = v(1:output_max_delay)';
B = v(output_max_delay+1:end)';

% Init predicted output
yh = zeros(1,N);

for i = 1:N
    
    % Predicted output
    yh(i) = A*Yd + B*Ud;
    
    % Uptade memory
    Ud = [u(i);Ud(1:end-1)];
    Yd = [y(i);Yd(1:end-1)];
    
end

%% Verify Prediction

figure; 
stem(n,yh,'.');
hold on
stem(n,y,'o');
hold off

MSE1 = (1/N)* sum((y-yh).^2);

%% Init Motor Estimation

clear;
clc;

%% Get Signals from Simulink

% Run Simulink File

simulation_duration = 10;
sim('dcMotor_DiagramBlocks.slx',simulation_duration)
% Tensao => Sample time: 0.001 | Noise power: 10 | Seed: [23342] |
% Carga =>  Sample time: 0.001 | Noise power: 02 | Seed: [23341] |

% With Voltage and Load

t = motor1.time';

Cl1 = motor1.signals.values(:,1)';
Ia1 = motor1.signals.values(:,2)';
Va1 = motor1.signals.values(:,3)';
Vel1 = motor1.signals.values(:,4)';

figure;
hold on
plot(t,Cl1,'y')
plot(t,Ia1,'b')
plot(t,Va1,'r')
plot(t,Vel1,'g')
title('Todas as Entradas')
hold off

% With Voltage = 0 and Load

t = motor2.time';

Cl2 = motor2.signals.values(:,1)';
Ia2 = motor2.signals.values(:,2)';
Va2 = motor2.signals.values(:,3)';
Vel2 = motor2.signals.values(:,4)';

figure;
hold on
plot(t,Cl2,'y')
plot(t,Ia2,'b')
plot(t,Va2,'r')
plot(t,Vel2,'g')
title('Tensão = 0')
hold off

% With Load = 0 and Voltage

t = motor3.time';

Cl3 = motor3.signals.values(:,1)';
Ia3 = motor3.signals.values(:,2)';
Va3 = motor3.signals.values(:,3)';
Vel3 = motor3.signals.values(:,4)';

figure;
hold on
plot(t,Cl3,'y')
plot(t,Ia3,'b')
plot(t,Va3,'r')
plot(t,Vel3,'g')
title('Carga = 0')
hold off

% Discrete Time

% t = motor4.time';
% 
% Cl4 = motor3.signals.values(:  ,1)';
% Ia4 = motor4.signals.values(:,2)'*0.001;
% Va4 = motor3.signals.values(:,3)';
% Vel4 = motor4.signals.values(:,1)'*0.001;
% 
% figure;
% hold on
% plot(t,Cl4,'y')
% plot(t,Ia4,'b')
% plot(t,Va4,'r')
% plot(t,Vel4,'g')
% title('Discrete Time')
% hold off

%% Define inputs and outputs

% W / Va transfer function
% u = Va3;
% y = Vel3;

% % Ia / Va transfer function
u = Va3;
y = Ia3;

% % W / Cl transfer function
% u = Cl2;
% y = Vel2;

% % Ia / Cl transfer function
% u = Cl2;
% y = Ia2;

%% Generate Prediction Vectors

N = length(y);
input_max_delay = 2;
output_max_delay = 2;

% X*v = e

X = zeros(N-output_max_delay,output_max_delay+input_max_delay);
e = zeros(N-output_max_delay,1);

for i = 1:(N-output_max_delay)
    e(i) = y(i+output_max_delay);
    X(i,:) = [ flip( y( i : (i+output_max_delay-1)) ) , ...
               flip( u( i : (i+input_max_delay-1) ) ) ];
end

%% Estimate Coefficients

v = ((X'*X)^(-1))*X'*e;

%% Predict 

N = length(y);
input_max_delay = 2;
output_max_delay = 2;

% Vectors with latest input and output samples

Ud = zeros(input_max_delay,1);
Yd = zeros(output_max_delay,1);

% Vectors with coefficients

A = v(1:output_max_delay)';
B = v(output_max_delay+1:end)';

% Init predicted output
yh = zeros(1,N);

for i = 1:N
    
    % Predicted output (model)
    yh(i) = A*Yd + B*Ud;
    
    % Uptade memory
    Ud = [u(i);Ud(1:end-1)];
    Yd = [y(i);Yd(1:end-1)];
    
end

%% Verify Prediction

figure;
plot(t,yh,'k.');
hold on
plot(t,y,'r-');
hold off

MSE2 = (1/N)* sum((y-yh).^2);

%% MISO SYSTEM - VELOCITY

% W x Va, Cl

%% MISO SYSTEM - CURRENT

% Ia x Va, Cl

%% END