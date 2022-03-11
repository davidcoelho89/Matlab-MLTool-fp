function Xlpc = linearPredictiveCoding(signals,Fs,Tqmin,Nquad,p)

% --- Build attributes matrix using the LPC method ---
%
%   Xlpc = linearPredictiveCoding(signals,Fs,Tqmin,Nquad,p)
%
%   Input:
%       signals = Cell with all signals        	[Ns x 1]
%       Fs = Sampling Frequency                 [cte] (Hz)
%       Tqmin = minimum duration of each frame  [cte] (s)
%       Nquad = Number of frames                [cte]
%       p = AR model order                      [cte]
%   Output:
%       Xlpc = Attributes Matrix                [Ns x Nquad*p]

%% INIT

Nsignals = length(signals);     % Numero de sinais
Nesp = Nquad-1;                 % Numero de espaços entre quadros
Xlpc = zeros(Nsignals,Nquad*p);	% Inicializa matriz de atributos

%% ALGORITHM

% Search for the biggest and the smallest signals

signal = signals{1};
Tmin = length(signal);
Tmax = length(signal);

for i = 2:Nsignals
    signal = signals{i};
    Tsig = length(signal);
    if (Tsig < Tmin)
        Tmin = Tsig;
    end
    if (Tsig > Tmax)
        Tmax = Tsig;
    end
end

% Compute AR(p) coefficients, per frame, for each signal

for i = 1:Nsignals
    
    % Get signal
    signal = signals{i};
    
    % Get auxiliary variables
    Tsig = length(signal);
    Tq = floor((Tsig*Tqmin*Fs)/Tmin);
    Tesp = floor((Tsig - Nquad*Tq)/Nesp);
    
    % Calcula modelo AR para cada quadro
    for j = 1:Nquad
        
        indices = (j-1)*(Tq+Tesp) + 1 : j*Tq + (j-1)*Tesp;
        quadro = signal( indices );
        
%         ah = ar_yw(quadro,p); % VERIFICAR IMPLEMENTACAO!
        
        ah = aryule(quadro,p);
        ah = -ah(2:end)';
        
        Xlpc(i,(j-1)*p+1:j*p) = ah;
        
        % Debug => verifica se modelo AR esta bem ajustado
%         if((i == 1) && (j == 1))
%             [X,y] = regressionMatrixFromTS(quadro,p);
%             yh = X*ah;
%             figure;
%             hold on
%             plot(1:length(y),y,'r-');
%             plot(1:length(yh),yh,'b-');
%             title('Modelagem de um quadro por modelo AR(10)')
%             ylabel('Amplitude')
%             xlabel('Amostras')
%             legend({'Sinal Original','Estimativa de Yule-Walker'}, ...
%                     'Location','northwest')
%             hold off
%         end
        
    end    

end

%% END


















