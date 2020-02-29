function Pout = ga_crossing(P,S,Pc,Fn,Nc,El)

% --- Crossing and Generate new Individuals ---
%
%   P = ga_crossing(P,S,Pc,Fn,Nc,El);
%
%   Inputs:
%       P = population of subjects                  	[Nc*Cl x Ni]
%       S = selected individuals for crossing           [1 x Ni]
%       Pc = Crossing probability                   	[cte]
%       Fn = Normalized fitness of each individual      [1 x N1]
%       Nc = Number of crhomossomes                     [cte]
%       El = Elitism                                    [cte]
%   Outputs:
%       Pout = new generation of subjects               [Nc*Cl x Ni]

%% INITIALIZATIONS

[Ngenes,Ni] = size(P);      % Number of genes and Number of individuals
Cl = Ngenes/Nc;             % Chromossomes Length

Pout = zeros(Ngenes,Ni);    % Next generation

%% ALGORITHM

for i = 1:(Ni/2),
    % Get parents
    p1 = P(:,S(2*i-1));
    p2 = P(:,S(2*i));
    % Init children
    f1 = p1;
    f2 = p2;
    % Aleatory generates number between 0 - 1
    u = rand;
    % Crossing over
    if (u <= Pc),
        % Defines "cut point"
        cut = floor((Cl-1)*rand) + 1;
        % Generate children
        for j = 1:Nc,
            f1((j-1)*Cl+1:j*Cl) = [p1((j-1)*Cl+1:(j-1)*Cl+cut) ; ...
                                   p2((j-1)*Cl+cut+1:j*Cl)];
            f2((j-1)*Cl+1:j*Cl) = [p2((j-1)*Cl+1:(j-1)*Cl+cut) ; ...
                                   p1((j-1)*Cl+cut+1:j*Cl)];
        end
    end
    % Fill new generation
    Pout(:,2*i-1) = f1;
    Pout(:,2*i) = f2;
end

% Elitism: keeps the fitest for the next generation
if (El == 1),
    [~,best] = max(Fn);
    Pout(:,1) = P(:,best);
end

%% FILL OUTPUT STRUCTURE



%% END