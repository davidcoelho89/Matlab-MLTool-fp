function [RC,RL,LC,sys] = filtpass_01(tipo,fc,fmin,fmax)
% 
% Função para gerar os valores de R*C e R/L de diverso tipos de filtros
% passivos
%
%   Entradas:
%       fc = frequência de corte do filtro
%       fmin e fmax = frequências para os filtros rejeita/passa faixa
%       tipo = tipo do filtro (PB_RC / PB_RL / PA_RC / PA_RL)
%                             (PF_S / PF_P / RF_S / RF_P)
%   Saídas
%       RC = valor de R*C
%       RL = valor de R/L
%       sys = função de transferência do filtro

Wc = 2*pi*fc;
Wmin = 2*pi*fmin;
Wmax = 2*pi*fmax;

switch upper(tipo)
    case 'PB_RC'
        RC = 1/Wc;
        RL = 0;
        LC = 0;
        num = [1];
        den = [RC 1];
        sys = tf(num,den);
    case 'PB_RL'
        RC = 0;
        RL = Wc;
        LC = 0;
        num = [RL];
        den = [1 RL];
        sys = tf(num,den);
    case 'PA_RC'
        RC = 1/Wc;
        RL = 0;
        LC = 0;
        num = [RC 0];
        den = [RC 1];
        sys = tf(num,den);
    case 'PA_RL'
        RC = 0;
        RL = Wc;
        LC = 0;
        num = [1 0];
        den = [1 RL];
        sys = tf(num,den);    
    case 'PF_S'
        RC = 0;
        RL = 2*pi*(fmax-fmin);
        LC = 1/(2*pi*fc)^2;
        num = [RL 0];
        den = [1 RL 1/LC];
        sys = tf(num,den);
    case 'PF_P'
        RC = 1/(2*pi*(fmax-fmin));
        RL = 0;
        LC = 1/(2*pi*fc)^2;
        num = [1/RC 0];
        den = [1 1/RC 1/LC];
        sys = tf(num,den);
    case 'RF_S'
        RC = 0;
        RL = 2*pi*(fmax-fmin);
        LC = 1/(2*pi*fc)^2;
        num = [1 0 1/LC];
        den = [1 RL 1/LC];
        sys = tf(num,den);
    case 'RF_P'
        RC = 1/(2*pi*(fmax-fmin));;
        RL = 0;
        LC = 1/(2*pi*fc)^2;
        num = [1 0 1/LC];
        den = [1 1/RC 1/LC];
        sys = tf(num,den);
    otherwise
        disp('Digite um tipo Válido')
end

figure; h = bodeplot(sys);
setoptions(h,'FreqUnits','Hz');
