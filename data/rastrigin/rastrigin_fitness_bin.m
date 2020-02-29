function [OUT] = rastrigin_fitness_bin(P)

% --- Rastrigin Binary Fitness Function ---
%
%   [out] = rastrigin_fitness(P)
%
%   Input:
%       P = population of subjects                          [Nc*Cl x Ni]
%   Output:
%       OUT.
%           X = phenotype of each individual                [Nc x Ni]
%           F = fitness of each individual                  [1 x N1]
%           Fn = Normalized fitness of each individual      [1 x N1]

%% INITIALIZATIONS

mind = [-5.12,-5.12];   % Minimum value of variables (in decimal)
maxd = [5.12,5.12];     % Maximum value of variables (in decimal)
Nc = 2;                 % number of chromossomes

[Ngenes,Ni] = size(P);  % Number of genes and Number of individuals
Cl = Ngenes/Nc;         % Chromossomes Length

X = zeros(Nc,Ni);   % Phenotype (real values of variables for each individual)
F = zeros(1,Ni);    % Fitness value for each individual

%% ALGORITHM

for i = 1:Ni,
    % Get subject
    subject = P(:,i);
    % Calculate Phenotype
    for j = 1:Nc,
        xj_bin = subject(Cl*(j-1)+1:Cl*(j));
        xj_dec = bin2deci(xj_bin);
        X(j,i) = mind(j) + (maxd(j)-mind(j))*xj_dec/(2^Cl - 1);
    end
    % Calculate Fitness of Subject
    F(i) = X(1,i)^2 + X(2,i)^2 + 20 ...
           - 10*(cos(2*pi*X(1,i))+ cos(2*pi*X(2,i)));
end

% Calculate Normalized Fitness
sig2n = 0.001;
Fn = (1./F + sig2n)/sum(1./F + sig2n);

%% FILL OUTPUT STRUCTURE

OUT.X = X;
OUT.F = F;
OUT.Fn = Fn;

%% END