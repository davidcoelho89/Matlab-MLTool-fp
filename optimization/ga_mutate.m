function Pout = ga_mutate(P,Pm,El)

% --- Mutate genes of Population ---
%
%   Pout = ga_mutate(P,Pm,Nc,El)
%
%   Input:
%       P = population of subjects       	[Nc*Cl x Ni]
%       Pm = Pm = Mutation probability      [cte]
%       El = Elitism                       	[cte]
%
%   Output:
%       Pout = new generation of subjects 	[Nc*Cl x Ni]

%% INITIALIZATIONS

[Ng,Ni] = size(P);
Pout = P;

%% ALGORITHM

for i = Ni,
    
    u = rand(Ng,1);     % generate vector with random numbers
    I = find(u <= Pm);  % mutate genes
    
    for j = 1:length(I),
        if Pout(I(j),i) == 0,
            Pout(I(j),i) = 1;
        else
            Pout(I(j),i) = 0;
        end
    end
end

% Elitism: keeps the fitest for the next generation
if(El == 1),
    Pout(:,1) = P(:,1);
end

%% FILL OUTPUT STRUCTURE



%% END