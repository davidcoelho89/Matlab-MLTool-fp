function [DATAout] = data_motor_gen(OPTION)

% --- Generates data for motor problem  ---
%
%   [DATAout] = data_motor_gen(OPTION)
%
%   Input:
%      OPTION.prob2 = Problem definition
%           (from 1 to 8 -> default classification)
%           01 = 2 classes: 42 N x 252 F
%           02 = 2 classes: 252 N x 252 F
%           03 = 7 classes: 294 samples - 42 per class
%           04 = 3 classes: 294 samples - 42 N x 126 AI x 126 BI
%           05 = 4 classes: 294 samples - 42 N x 84 nv1 x 84 nv2 x 84 nv3
%           06 = 7 classes: 504 samples - 252 N x 42 per class of fault
%           07 = 3 classes: 504 samples - 252 N x 126 AI x 126 BI
%           08 = 4 classes: 504 samples - 252 N x 84 nv1 x 84 nv2 x 84 nv3
%           (from 9 to end -> ordinal classification)
%           09 = 2 classes: 252 N x 252 F ()
%           10 = 2 classes: 252 N x 252 F ()
%           11 = 2 classes: 252 N x 252 F ()
%           12 = 2 classes: 252 N x 252 F ()
%       OPTION.prob2 = More details about a specific data set
%   Output:
%       DATAout = general data
%           .input   = attributes' matrix [pxN]
%           .output  = labels' matrix [1xN]
%                      (with just 1 value - 1 to Nc)
%                      (includes ordinal classification) 
%           .lbl     = labels' vector [1xN]
%                      (original labels of data set)

%% INITIALIZATIONS

problem = OPTION.prob2;

%% ALGORITHM

switch(problem)

%---------- BINARY PROBLEM - 42 X 252 -----------%

case {1},

load data_ESPEC_1.mat

input = data1;          % Input initialization
[~,N] = size(input);    % Number of samples
output = zeros(1,N);    % Output initialization
lbl = rot + 1;          % Original Labels

for i = 1:N,
if lbl(i) == 1,
    output(i) = 1;
else
    output(i) = 2;
end
end

%--------- BINARY PROBLEM - 252 X 252 -----------%

case {2},

load data_ESPEC_5.mat

input = data3;          % Input initialization
[~,N] = size(input);    % Number of samples
output = zeros(1,N);    % Output initialization
lbl = rot3 + 1;         % Original Labels

for i = 1:N,
if lbl(i) == 1,
    output(i) = 1;
else
    output(i) = 2;
end
end

%------- PROBLEMA MULTICLASSES - 6 CLASSES -------%

case {3},

load data_ESPEC_1.mat

input = data1;          % Input initialization
lbl = rot + 1;          % Original Labels
output = lbl;           % Standard Output

%------ PROBLEMA MULTICLASSES - N x AI x BI ------%

case {4},

load data_ESPEC_1.mat

input = data1;          % Input initialization
[~,N] = size(input);    % Number of samples
output = zeros(1,N);    % Output initialization
lbl = rot + 1;          % Original Labels

for i = 1:N,
    % Normal
    if (lbl(i) == 1),
        output(i) = 1;
    % High Impedance
    elseif (lbl(i) == 2 || lbl(i) == 3 || lbl(i) == 4),
        output(i) = 2;
    % Low Impedance
    elseif (lbl(i) == 5 || lbl(i) == 6 || lbl(i) == 7),
        output(i) = 3;
    end        
end
    
%-- PROBLEMA MULTICLASSES - N x nv1 x nv2 x nv3 --%

case {5},

load data_ESPEC_1.mat

input = data1;          % Input initialization
[~,N] = size(input);    % Number of samples
output = zeros(1,N);    % Output initialization
lbl = rot + 1;          % Original Labels

for i = 1:N,
    % Normal
    if (lbl(i) == 1),
        output(i) = 1;
    % lvl1
    elseif (lbl(i) == 2 || lbl(i) == 5),
        output(i) = 2;
    % lvl2
    elseif (lbl(i) == 3 || lbl(i) == 6),
        output(i) = 3;
    % lvl3
    elseif (lbl(i) == 4 || lbl(i) == 7),
        output(i) = 4;
    end
end

%------- PROBLEMA MULTICLASSES - 6 CLASSES -------%

case {6},
    
load data_ESPEC_5.mat

input = data3;          % Input initialization
lbl = rot3 + 1;         % Original Labels
output = lbl;           % Standard Output

%------ PROBLEMA MULTICLASSES - N x AI x BI ------%

case {7},

load data_ESPEC_5.mat

input = data3;          % Input initialization
[~,N] = size(input);    % Number of samples
output = zeros(1,N);    % Output initialization
lbl = rot3 + 1;         % Original Labels

for i = 1:N,
    % Normal
    if (lbl(i) == 1),
        output(i) = 1;
    % High Impedance
    elseif (lbl(i) == 2 || lbl(i) == 3 || lbl(i) == 4),
        output(i) = 2;
    % Low Impedance
    elseif (lbl(i) == 5 || lbl(i) == 6 || lbl(i) == 7),
        output(i) = 3;
    end        
end   
    
%-- PROBLEMA MULTICLASSES - N x nv1 x nv2 x nv3 --%

case {8},

load data_ESPEC_5.mat

input = data3;          % Input initialization
[~,N] = size(input);    % Number of samples
output = zeros(1,N);    % Output initialization
lbl = rot3 + 1;         % Original Labels

for i = 1:N,
    % Normal
    if (lbl(i) == 1),
        output(i) = 1;
    % lvl1
    elseif (lbl(i) == 2 || lbl(i) == 5),
        output(i) = 2;
    % lvl2
    elseif (lbl(i) == 3 || lbl(i) == 6),
        output(i) = 3;
    % lvl3
    elseif (lbl(i) == 4 || lbl(i) == 7),
        output(i) = 4;
    end
end

%------ PROBLEMA BINARIO - CLASS ORDINAL 1 -------%
    
case {9},

% Neste, o nível da gradação é feito da seguinte forma:
% Classes: 0   1    2    3    4    5   6
% Rotulos: 1 -0.3 -0.4 -0.6 -0.7 -0.8 -1
% nivel de falha aumentando linearmente

% VERIFICAR "CASE 7"

load data_ESPEC_5.mat

% padrão [pxN] = [Atributo x Amostra]

input = data;           % Inicialização dos dados
[~,N] = size(input);

% padrão [cxN] = [Classe x Amostra]

output = -1*ones(Nc,N);  % Inicialização dos alvos

lbl = lbl + 1;
for i = 1:N,
    if (lbl(i) == 1),
        output(1,i) = 1;
    elseif (lbl(i) == 2),
        output(1,i) = -0.3;
        output(2,i) = 0.3;
    elseif (lbl(i) == 3),
        output(1,i) = -0.4;
        output(2,i) = 0.4;
    elseif (lbl(i) == 4),
        output(1,i) = -0.6;
        output(2,i) = 0.6;
    elseif (lbl(i) == 5),
        output(1,i) = -0.7;
        output(2,i) = 0.7;
    elseif (lbl(i) == 6),
        output(1,i) = -0.8;
        output(2,i) = 0.8;
    elseif (lbl(i) == 7),
        output(1,i) = -1;
        output(2,i) = 1;
    end
end

%------ PROBLEMA BINARIO - CLASS ORDINAL 2 -------%

case {10},

% Neste, o nível da gradação é feito da seguinte forma:
% Classes: 0   1    2    3    4    5   6
% Rotulos: 1 -0.3 -0.5 -0.8 -0.4 -0.6 -0.9
% Nivel da falha aumenta com a qtdade de espiras em curto
% 1 e 4 (1.41%) / 2 e 5 (4.84%) / 3 e 6 (9.86%)

% VERIFICAR "CASE 7"

load data_ESPEC_5.mat

% padrão [pxN] = [Atributo x Amostra]

input = data;           % Inicialização dos dados
[~,N] = size(input);

% padrão [cxN] = [Classe x Amostra]

output = -1*ones(Nc,N);  % Inicialização dos alvos

lbl = lbl + 1;
for i = 1:N,
    if (lbl(i) == 1),
        output(1,i) = 1;
    elseif (lbl(i) == 2),
        output(1,i) = -0.3;
        output(2,i) = 0.3;
    elseif (lbl(i) == 3),
        output(1,i) = -0.5;
        output(2,i) = 0.5;
    elseif (lbl(i) == 4),
        output(1,i) = -0.8;
        output(2,i) = 0.8;
    elseif (lbl(i) == 5),
        output(1,i) = -0.4;
        output(2,i) = 0.4;
    elseif (lbl(i) == 6),
        output(1,i) = -0.6;
        output(2,i) = 0.6;
    elseif (lbl(i) == 7),
        output(1,i) = -0.9;
        output(2,i) = 0.9;
    end
end

%------ PROBLEMA BINARIO - CLASS ORDINAL 3 -------%

case {11},

% Neste, o nível da gradação é feito da seguinte forma:
% Classes: 0   1    2    3    4    5   6
% Rotulos: 1 -0.5 -0.8 -0.9 -0.6 -0.85 -0.95
% Nivel da falha aumenta com a qtdade de espiras em curto
% 1 e 4 (1.41%) / 2 e 5 (4.84%) / 3 e 6 (9.86%)

% VERIFICAR "CASE 7"

load data_ESPEC_5.mat

% padrão [pxN] = [Atributo x Amostra]

input = data;           % Inicialização dos dados
[~,N] = size(input);

% padrão [cxN] = [Classe x Amostra]

output = -1*ones(Nc,N);  % Inicialização dos alvos

lbl = lbl + 1;
for i = 1:N,
    if (lbl(i) == 1),
        output(1,i) = 1;
    elseif (lbl(i) == 2),
        output(1,i) = -0.5;
        output(2,i) = 0.5;
    elseif (lbl(i) == 3),
        output(1,i) = -0.8;
        output(2,i) = 0.8;
    elseif (lbl(i) == 4),
        output(1,i) = -0.9;
        output(2,i) = 0.9;
    elseif (lbl(i) == 5),
        output(1,i) = -0.6;
        output(2,i) = 0.6;
    elseif (lbl(i) == 6),
        output(1,i) = -0.85;
        output(2,i) = 0.85;
    elseif (lbl(i) == 7),
        output(1,i) = -0.95;
        output(2,i) = 0.95;
    end
end

%------ PROBLEMA BINARIO - CLASS ORDINAL 4 -------%

case {12},

% Neste, o nível da gradação é feito da seguinte forma:
% Classes: 0   1    2    3    4    5   6
% Rotulos: 1 -0.5 -0.6 -0.8 -0.85 -0.9 -0.95
% nivel de falha aumentando linearmente

% VERIFICAR "CASE 7"

load data_ESPEC_5.mat

% padrão [pxN] = [Atributo x Amostra]

input = data;           % Inicialização dos dados
[~,N] = size(input);

% padrão [cxN] = [Classe x Amostra]

output = -1*ones(Nc,N);  % Inicialização dos alvos

lbl = lbl + 1;
for i = 1:N,
    if (lbl(i) == 1),
        output(1,i) = 1;
    elseif (lbl(i) == 2),
        output(1,i) = -0.5;
        output(2,i) = 0.5;
    elseif (lbl(i) == 3),
        output(1,i) = -0.6;
        output(2,i) = 0.6;
    elseif (lbl(i) == 4),
        output(1,i) = -0.8;
        output(2,i) = 0.8;
    elseif (lbl(i) == 5),
        output(1,i) = -0.85;
        output(2,i) = 0.85;
    elseif (lbl(i) == 6),
        output(1,i) = -0.9;
        output(2,i) = 0.9;
    elseif (lbl(i) == 7),
        output(1,i) = -0.95;
        output(2,i) = 0.95;
    end
end

%--------- NENHUMA DAS OPÇÕES ANTERIORES ---------%

otherwise,

input = [];
output = [];
lbl = [];
disp('Choose a correct option. Void data was generated.')

%---------------------TERMINO---------------------%

end

%% FILL OUTPUT STRUCTURE

DATAout.input = input;
DATAout.output = output;
DATAout.lbl = lbl;

%% END