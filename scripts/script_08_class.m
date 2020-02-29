%% CLASSIFICAÇÃO - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

nfig = 0;       % contagem do numero de figuras

%% DISTANCIAS (normas)

% pdist2();               % distância quadrática
% norm();                 % varias distâncias (normas)

%% ORGANIZAR DADOS

% sort();    	% Organiza em ordem crescente ou decrescente

% bubble sort   % Organizacao por borbulhamento
                % Colocar criterio de parada

% matc = importdata('Derma.m');
% %vetor auxiliar
% vetaux = zeros(1,35);
% cont2 = 0;
% cont3 = 0;
% for i = 1:365,
%     for j = 1:365,
%         if (matc(j,35) > matc(j+1,35)),
%             vetaux = matc(j,:);
%             matc(j,:) = matc(j+1,:);
%             matc(j+1,:) = vetaux;
%         end
%     end
% end

%% FUNÇÕES GERAIS

X = [1 1 -1 -1 1 1 1]'; % inicializa vetor de rotulos
tabulate(X);            % porcentagem de cada rotulo

%% END
