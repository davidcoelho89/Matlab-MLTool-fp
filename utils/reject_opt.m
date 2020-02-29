function [OUT] = reject_opt(DATA,OUTts,REJp)

% --- Classifier's Reject Option ---
%
%   [OUTrj] = reject_opt(DATA,OUTts,REJp,OPTION)
%
%   Input:
%       DATA.
%           output = output matrix                              [Nc x Nts]
%       OUTts.
%           y_h = estimated output matrix                       [Nc x Nts]
%       REJp.
%           band = range of values to rejected                  [cte]
%           prob = kind of problem                              [cte]
%   Output:
%       rejected = Matrix indicating rejected samples           [Nr x 2]
%           rejected(r,1) = yh
%           rejected(r,2) = y
%       reject_ratio = Nr / Nts                                 [0 - 1]
%       acc = accuracy rate without rejected samples            [0 - 1]

%% INICIALIZAÇÕES

problem = REJp.prob;        % problem definition
reject_band = REJp.band;    % threshold of rejection
yh = OUTts.y_h;             % classifier output
y = DATA.output;            % real output
[Nc,tst] = size(y);         % Number of classes and samples

% Auxiliary Variables

y_seq = zeros(1,tst);       % Real sequential label
yh_seq = zeros(1,tst);      % Estimated sequential label

% Inicializa Saidas

n_rej = length(find(max(yh) < reject_band));
rejected = cell(n_rej,2);  % stores rejected samples

%% CONFIGURA ALVOS

% Configura labels de saída

switch problem,
    case 1,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 2,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 3,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 4,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 5,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 6,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 7,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 8,
        Mconf = zeros(Nc); % Matriz de confusao
        [~,y_seq] = max(y);
        [~,yh_seq] = max(yh);
    case 9,
        Mconf = zeros(7); % Matriz de confusao
        for i = 1:tst,
            if(yh(1,i) >= 0),
                yh_seq(i) = 1;
            elseif ((yh(1,i) < 0) && (yh(1,i) >= -0.3))
                yh_seq(i) = 2;
            elseif ((yh(1,i) < -0.3) && (yh(1,i) >= -0.4))
                yh_seq(i) = 5;
            elseif ((yh(1,i) < -0.4) && (yh(1,i) >= -0.5))
                yh_seq(i) = 3;
            elseif ((yh(1,i) < -0.5) && (yh(1,i) >= -0.6))
                yh_seq(i) = 6;
            elseif ((yh(1,i) < -0.6) && (yh(1,i) >= -0.8))
                yh_seq(i) = 4;
            elseif (yh(1,i) < -0.8)
                yh_seq(i) = 7;
            end
            
            switch y(1,i),
                case 1
                    y_seq(i) = 1;
                case -0.3
                    y_seq(i) = 2;
                case -0.5
                    y_seq(i) = 3;
                case -0.8
                    y_seq(i) = 4;
                case -0.4
                    y_seq(i) = 5;
                case -0.6
                    y_seq(i) = 6;
                case -0.9
                    y_seq(i) = 7;
            end
        end
        
    case 10,
        Mconf = zeros(7); % Matriz de confusao
        for i = 1:tst,
            if(yh(1,i) >= 0),
                yh_seq(i) = 1;
            elseif ((yh(1,i) < 0) && (yh(1,i) >= -0.5))
                yh_seq(i) = 2;
            elseif ((yh(1,i) < -0.5) && (yh(1,i) >= -0.6))
                yh_seq(i) = 5;
            elseif ((yh(1,i) < -0.6) && (yh(1,i) >= -0.8))
                yh_seq(i) = 3;
            elseif ((yh(1,i) < -0.8) && (yh(1,i) >= -0.85))
                yh_seq(i) = 6;
            elseif ((yh(1,i) < -0.85) && (yh(1,i) >= -0.9))
                yh_seq(i) = 4;
            elseif (yh(1,i) < -0.9)
                yh_seq(i) = 7;
            end
            
            switch y(1,i),
                case 1
                    y_seq(i) = 1;
                case -0.5
                    y_seq(i) = 2;
                case -0.8
                    y_seq(i) = 3;
                case -0.9
                    y_seq(i) = 4;
                case -0.6
                    y_seq(i) = 5;
                case -0.85
                    y_seq(i) = 6;
                case -0.95
                    y_seq(i) = 7;
            end            
        end
        
    case 11,
        Mconf = zeros(7); % Matriz de confusao
        for i = 1:tst,
            if(yh(1,i) >= 0),
                yh_seq(i) = 1;
            elseif ((yh(1,i) < 0) && (yh(1,i) >= -0.5))
                yh_seq(i) = 2;
            elseif ((yh(1,i) < -0.5) && (yh(1,i) >= -0.6))
                yh_seq(i) = 3;
            elseif ((yh(1,i) < -0.6) && (yh(1,i) >= -0.8))
                yh_seq(i) = 4;
            elseif ((yh(1,i) < -0.8) && (yh(1,i) >= -0.85))
                yh_seq(i) = 5;
            elseif ((yh(1,i) < -0.85) && (yh(1,i) >= -0.9))
                yh_seq(i) = 6;
            elseif (yh(1,i) < -0.9)
                yh_seq(i) = 7;
            end
            
            switch y(1,i),
                case 1
                    y_seq(i) = 1;
                case -0.5
                    y_seq(i) = 2;
                case -0.6
                    y_seq(i) = 3;
                case -0.8
                    y_seq(i) = 4;
                case -0.85
                    y_seq(i) = 5;
                case -0.9
                    y_seq(i) = 6;
                case -0.95
                    y_seq(i) = 7;
            end                
        end
        
    case 12,
        Mconf = zeros(7); % Matriz de confusao

    otherwise
        disp('Wrong Problem Type')
end

%% ALGORITMO

n_rej = 0;

for i = 1:tst,
    if (max(yh(:,i)) < reject_band),
        n_rej = n_rej+1;
        rejected{n_rej,1} = yh(:,i);
        rejected{n_rej,2} = y(:,i);
    else
        Mconf(y_seq(i),yh_seq(i)) = Mconf(y_seq(i),yh_seq(i)) + 1;
    end
end

% Calcula razao de amostras rejeitadas e taxa de acerto

reject_ratio = n_rej/tst;

if (reject_ratio == 1),
    acerto = -1;
else
    acerto = (sum(diag(Mconf)))/tst;
end

%% FILL OUTPUT STRUCTURE

OUT.rejected = rejected;
OUT.reject_ratio = reject_ratio;
OUT.Mconf = Mconf;
OUT.acc = acerto;

%% END