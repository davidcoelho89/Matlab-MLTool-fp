%% PCA - SFS e SBS

% SFS (Sequential Foward Selection)

% inicializa sem nenhum atributo.
% adiciona sequencialmente os atributos (2x for 1:p)
%   verifica se atributo foi selecionado
%   caso negativo, adiciona momentaneamente este a matriz
%   aplica pca nesta matriz momentanea    
%   divide dados entre treino e teste
%   aplica um classificador
%   calcula função custo = accuracy (pode repetir isso 20x -> mean)
%   adiciona o atributo que obteve o melhor custo permanentemente à matriz

% ToDo - All

% SBS (Sequential Backward Selection)

% inicializa com todos os atributos
% vai retirando sequencialmente os atributos (2x for 1:p)
% (mesma ideia do SFS)

% ToDo - All

%% END