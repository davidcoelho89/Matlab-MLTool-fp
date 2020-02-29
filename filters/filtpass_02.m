function [sys] = filtpass_02(tipo,R,L,C)
% 
% Função para gerar a resposta de um filtro passivo, dado os valores comerciais
% de R, L, C, e o tipo do filtro.
%
%   Entradas:
%       R / L / C = valor dos resistor, indutor e capacitor.
%       tipo = tipo do filtro (PB_RC / PB_RL / PA_RC / PA_RL)
%                             (PF_S / PF_P / RF_S / RF_P)
%   Saídas
%       sys = função de transferência do filtro

switch upper(tipo)
    case 'PB_RC'
        num = [1];
        den = [R*C 1];
        sys = tf(num,den);
    case 'PB_RL'
        num = [R/L];
        den = [1 R/L];
        sys = tf(num,den);
    case 'PA_RC'
        num = [R*C 0];
        den = [R*C 1];
        sys = tf(num,den);
    case 'PA_RL'
        num = [1 0];
        den = [1 R/L];
        sys = tf(num,den);
    case 'PF_S'
        num = [R/L 0];
        den = [1 R/L 1/(L*C)];
        sys = tf(num,den);
    case 'PF_P'
        num = [1/(R*C) 0];
        den = [1 1/(R*C) 1/(L*C)];
        sys = tf(num,den);
    case 'RF_S'
        num = [1 0 1/(L*C)];
        den = [1 1/(R*C) 1/L*C];
        sys = tf(num,den);
    case 'RF_P'
        num = [1 0 1/(L*C)];
        den = [1 1/(R*C) 1/(L*C)];
        sys = tf(num,den);
    otherwise
        disp('Digite um tipo Válido')
end

figure; h = bodeplot(sys);
setoptions(h,'FreqUnits','Hz'); %plotar bode em Hz
