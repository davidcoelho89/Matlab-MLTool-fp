%% init

clear;
close;
clc;

%% Experiment 1: Continuous and Discrete Systems Simulation

% Simulando um sistema continuo

num = [0 1];
den = [1 1];
sys = tf(num,den);
[h,t] = impulse(sys);

figure;
set(gca,'FontSize',18)
plot(t,h)
title('Impulse Response')
ylabel('h(t)')
xlabel('t (s)')

t = 0:0.01:20;
w0 = 3;
x = cos(w0*t);

y = lsim(sys,x,t); % Saida para uma entrada senoidal

figure;
set(gca,'FontSize',18)
plot(t(1:500),x(1:500),'b',t(1:500),y(1:500),'r')
grid on
ylabel('x(t) e y(t)')
xlabel('t (s)')
axis([0 5 -1.2 1.2])
legend('x(t)','y(t)')

% Simulando um sistema discreto

k = 1:1000;
w0 = pi/10;
x = cos(w0*k);

y = zeros(1,length(k));
for n = 3:length(k);
    y(n) = (3/4)*y(n-1) - (1/8)*y(n-2) + 2*x(n);
end

figure;
set(gca,'FontSize',18)
stem(k(1:50),x(1:50),'b');
hold on
stem(k(1:50),y(1:50),'r');
hold off

%% Experiment 2:



%% End