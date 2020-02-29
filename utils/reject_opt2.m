function [OUT] = reject_opt2(DATA,OUT,REJp)

% --- Classifier's Reject Option ---
%
%   [OUT] = reject_opt2(DATAts,OUTts,REJp)
%
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       OUT.
%           y_h = estimated output                              [Nc x N]
%       REJp.
%           wr = Rejection cost                                 [cte]
%   Output:
%       OUT.
%           R = rejected pattern rate                        	[0 - 1]
%           E = classification errors                          	[0 - 1]
%           B = optimal rejection threshold                    	[0 - 1]
%           Mconf = new confusion matrix                       	[Nc x Nc]
%           acc = new accuracy rate
%           rej = matrix with rejected samples               	[Nc x Nr]

%% INICIALIZAÇÕES

% Inicializa Entradas

wr = REJp.w;           	% Rejection cost
Yh = OUT.y_h;        	% Classifiers outputs
Y = DATA.input;         % Real outputs
[Nc,N] = size(Y);       % Number of samples and classes

% Inicializa Saidas

Mconf = zeros(Nc,Nc); 	% New confusion matrix

% Rejection Threshold Variation

Bi = 0.25;
dB = 0.05;
Bf = 1.00;

% Init Empirical risk (for each rejection threshold)

RJ_threshold = Bi:dB:Bf;
RJ_length = length(RJ_threshold);

Rh = zeros(1,RJ_length);

%% ALGORITMO

% Find optimal threshold

j = 0;      % index of R_hatched

for Btst = Bi:dB:Bf,

Nr = 0;     % no of rejected patterns
Ne = 0;     % no of classification errors
j = j+1;    % increment index

for i = 1:N,
    % If output < threshold: increment number of rejected patterns
    if (max(Yh(:,i)) < Btst),
        Nr = Nr+1;
    else
        [~,y] = max(Y(:,i));
        [~,yh] = max(Yh(:,i));
        % If there is an error, increment number of misclassified patterns
        if (y ~= yh)
            Ne = Ne+1;
        end
    end
end

Rb = Nr/N;          % Rejection rate for specific threshold
Eb = Ne/(N - Nr);   % Misclassification rate for specific threshold

Rh(j) = wr*Rb + Eb; % Empirical Risk

end

Bo = min(Rh);

% Save rejected samples

Nr = length(find(max(Yh) < Bo));    % no of rejected samples
rejected = cell(Nr,2);          	% save rejected samples

% Calculate outputs

r = 0;     	% rejected pattern
Ne = 0;  	% no of misclassified patterns

for i = 1:N,
   	% se saida menor que limiar, Incrementa no de padroes rejeitados
    if (max(Yh(:,i)) < Bo),
        r = r + 1;
        rejected{r,1} = Yh(:,i);
        rejected{r,2} = Y(:,i);
    else
        [~,y] = max(Y(:,i));
        [~,yh] = max(Yh(:,i));
        % New confusion matrix
        Mconf(y,yh) = Mconf(y,yh) + 1;
        if (y ~= yh)
            Ne = Ne + 1;
        end
    end
end

R = Nr/N;           % taxa de rejeicao
E = Ne/(N - Nr);    % erros de classificacao
B = Bo;             % limiar de rejeicao

% Nova taxa de acerto

if (Bo == 1),
    acc = -1;
else
    acc = (sum(diag(Mconf)))/N;
end

%% FILL OUTPUT STRUCTURE

OUT.R = R;
OUT.E = E;
OUT.B = B;
OUT.Mconf = Mconf;
OUT.acc = acc;
OUT.rej = rejected;

%% END