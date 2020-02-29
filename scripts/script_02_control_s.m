%% CONTROLE - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

nfig = 0;   % contagem do numero de figuras

%% Sistema de 1a Ordem

T = 1;  % Constante de tempo

a0 = 1;
b1 = T;
b0 = 1;

num1 = [0 a0];          % numerador do sistema de 1a ordem
den1 = [b1 b0];         % denominador do sistema de 1a ordem

sys1 = tf(num1,den1);   % função de transferência 1

%% Sistema de 2a Ordem

psi = 0.3;  % Taxa de amortecimento
Wn = 3;     % Frequência Natural

a0 = Wn^2;
b2 = 1;
b1 = 2*psi*Wn;
b0 = Wn^2;

num2 = [0 0 a0];    % numerador do sistema de 2a ordem
den2 = [b2 b1 b0];  % denominador do sistema de 2a ordem

sys2 = tf(num2,den2);   % função de transferência 2

%% Pólos dos sistemas

raizes1 = roots(den1);  % raizes do denominador
raizes2 = roots(den2);  % raizes do denominador

%% Frações Parciais

num = [0 0 1];
den = [1 2 1];

% r1/(s-p1) + r2/(s-p2) + ... + k
[r,p,k] = residue(num,den);

%% Aplicação de um impulso

nfig = nfig + 1;    figure(nfig)

impulse(sys1)       % impulso a um sistema de 1a ordem

nfig = nfig + 1;    figure(nfig)

impulse(sys2)       % impulso a um sistema de 2a ordem

%% Aplicação de um degrau

nfig = nfig + 1;    figure(nfig)

step(sys1);         % degrau a um sistema de 1a ordem

nfig = nfig + 1;    figure(nfig)

[y,t] = step(sys1); % outra forma de plotar 
plot(t,y);          % (sai sem as labels)

nfig = nfig + 1;    figure(nfig)

step(sys2);         % degrau a um sistema de 2a ordem

%% Aplicação de uma rampa

% Primeiro multiplica o sist. por 1/s , depois aplica um degrau 1/s
% (é o mesmo que aplicar uma rampa diretamente (1/s^2)

den3 = [den2 0];
sys3 = tf(num2,den3);

nfig = nfig + 1;    figure(nfig)

step(sys3);         % rampa a um sistema de 2a ordem

%% Lugar das Raizes

% função na forma K*[num]/[den] = -1 

nfig = nfig + 1;    figure(nfig)

rlocus(sys2);

%% Diagrama de Bode

nfig = nfig + 1;    figure(nfig)

Kp = 2; Ki = 1; Kd = 1;         % Ganhos do controlador

A1 = [51529];                   % numerador em malha aberta
B1 = [1 2333.4 7470.5 0];       % denominador em malha aberta

A2 = [Kp * 51529];              % numerador em malhar fechada
B2 = [1 2333.4 7470.5 51529];   % denominador em malha fechada

sys1 = tf(A1,B1);               
bode(sys1,'b');                 % diagrama de bode

hold on

sys2 = tf(A2,B2);               
bode(sys2,'r');                 % diagrama de bode

hold off

% Margens de Ganho e de Fase

dc1 = dcgain(sys1);             % ganho dc de malha aberta
dc2 = dcgain(sys2);             % ganho dc de malha fechada

% margin(sys)

%% Diagrama de Nyquist

% nyquist(sys)

%% Aplicando PID

Kp = 4; Ki = 50; Kd = 1.9;

A1 = [51529];
B1 = [1 2333.4 7470.5 51529];

P = tf(A1,B1);
C = pid(Kp, Ki, Kd);
T = feedback(C * P, 1);

t = 0:0.01:2;

nfig = nfig + 1;    figure(nfig)

step(P);
stepinfo(P)

nfig = nfig + 1;    figure(nfig)

step(T);
stepinfo(T)

%% MOSTRAR TODOS OS PARAMETROS DO SISTEMA

A = [51529];
B = [1 2333.4 7470.5 51529];

disp('O sistema é:')
sys = tf(A, B)

disp('A ordem do sistema é:')
order = order(sys)

disp('Os zeros do sistema são:')
zero(sys)

disp('Os pólos do sistema são:')
poles = pole(sys)

disp('O ganho DC do sistema é:')
Kg = dcgain(sys)

disp('O sistema é estável?')
isstable(sys)

disp('O diagrama de pólos e zeros é:')
nfig = nfig + 1;    figure(nfig)
pzmap(sys)

disp('A resposta impulsiva do sistema é:')
nfig = nfig + 1;    figure(nfig)
impulse(sys)

disp('A resposta ao degrau do sistema é:')
nfig = nfig + 1;    figure(nfig)
step(sys)

disp('Conclusões sobre a resposta ao degrau:')
stepinfo(sys)

disp('Diagrama de Bode:')
nfig = nfig + 1;    figure(nfig)
bode(sys);
 
disp('Diagrama do Lugar das Raízes:')
nfig = nfig + 1;    figure(nfig)
rlocus(sys);

disp('Diagrama de Nyquist:')
nfig = nfig + 1;    figure(nfig)
nyquist(sys)

disp('O sistema inverso é:')
inv(sys)
pause();
close all;

%% CRITERIO DE ESTABILIDADE DE ROUTH

clc; clear;
r = input('input vector of your system coefficents: ');
m = length(r);
n = round(m/2);
q = 1;
k = 0;

for p = 1:length(r) 
    if rem(p,2)==0
        c_even(k)=r(p); 
    else
        c_odd(q)=r(p); 

        k=k+1;
        q=q+1;
    end
end

a=zeros(m,n);

if m/2 ~= round(m/2)
    c_even(n)=0;
end
a(1,:)=c_odd;
a(2,:)=c_even;
if a(2,1)==0
    a(2,1)=0.01;
end
for i=3:m
    for j=1:n-1
        x=a(i-1,1);
        if x==0
            x=0.01;
        end

        a(i,j)=((a(i-1,1)*a(i-2,j+1))-(a(i-2,1)*a(i-1,j+1)))/x;

    end
    if a(i,:)==0
        order=(m-i+1);
        c=0;
        d=1;
        for j=1:n-1
            a(i,j)=(order-c)*(a(i-1,d));
            d=d+1;
            c=c+2;
        end
    end
    if a(i,1)==0
        a(i,1)=0.01;
    end
end
Right_poles=0;
for i=1:m-1
    if sign(a(i,1))*sign(a(i+1,1))==-1
        Right_poles=Right_poles+1;
    end
end
fprintf('\n Routh-Hurwitz Table:\n')
a
fprintf('\n Number Of Right Poles =%2.0f\n',Right_poles)

reply = input('Do You Need Roots of System? Y/N ', 's');
if reply=='y'||reply=='Y'
    ROOTS = roots(r);
    fprintf('\n Given Polynomials Coefficents Roots :\n')
    ROOTS
else
    
end

%% END
