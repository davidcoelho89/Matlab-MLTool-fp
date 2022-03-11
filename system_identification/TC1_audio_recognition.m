%% IDENTIFICACAO DE SISTEMAS

% TC1 - Reconhecimento de Comandos de Voz
% Autor: David Nascimento Coelho
% Data: 24/02/2022

close;
clear;
clc;

%% (0) Add Folder and Subfolders to Matlab Path

% Get full path of current file
filePath = matlab.desktop.editor.getActiveFilename;

% Use the right symbol for the path                                                                           
if(isunix)   
    symb = '/';                   
else
    symb = '\'; 
end

% Find position of last "\" or "/"
nchar = length(filePath);
for i = 1:nchar
    if(filePath(i) == symb)
        lastSymb = i;
    end
end

folderPath = filePath(1:lastSymb);

% Find position of penultimate "\" or "/"
nchar = length(folderPath);
nchar = nchar-1;
for i = 1:nchar
    if(folderPath(i) == symb)
        lastSymb = i;
    end
end

upperfolderPath = folderPath(1:lastSymb);

% Add folder and subfolders to the path
addpath(genpath(upperfolderPath));

% Clear variables
clear filePath folderPath upperfolderPath i lastSymb nchar symb

%% (1) Load Audio Files

% 5 comandos de audio, 10 repetições de cada comando
% 2 audios por arquivo (stereo), 2 subamostras por audio
% Freq original: 44100 Hz

sub_sampling = 2;   % Generate more samples from one signal (If > 1)
mean_rem = 1;       % Remove mean from signals (If ~= 0)

[signals,classes,~,Fs] = loadAudioData(sub_sampling,mean_rem);

%% (2) Build Samples with LPC Method

Tqmin = 0.015;      % Tamanho mínimo de um quadro: 15 ms
Nquad = 31;         % Numero de quadros
p = 10;             % Ordem do modelo AR(p)

Xlpc = linearPredictiveCoding(signals,Fs,Tqmin,Nquad,p);

DATAlpc.input = Xlpc';
DATAlpc.output = classes';
save('DATAlpc.mat','DATAlpc');

%% (3) Build Samples with PSD Method

Xfreq = powerSpectralDensity(signals,Fs);

DATApsd.input = Xfreq';
DATApsd.output = classes';
save('DATApsd.mat','DATApsd');

%% (4) Funções Implementadas

% ols_train
% ols_classify
% "Box-Cox": Xbc = (X^gamma - 1)/gamma
% normalize_fit
% normalize transform
% mlp_train
% mlp_classify
% pca_feature
% pca_transform

%% (5) Compararação de Desempenho

audioRecognition_pipeline

%% (6.1) Adição de Ruído e Filtro PB - Gera sinais

% Init
close;
clear;
clc;

% Load Audio Files
sub_sampling = 2;   % Generate more samples from one signal (If > 1)
mean_rem = 1;       % Remove mean from signals (If ~= 0)
[signals,classes,~,Fs] = loadAudioData(sub_sampling,mean_rem);

% See original signal
signal = signals{1};
figure;
plot(signal);
title('Sinal Original - Comando Avançar');
xlabel('Amostras');
ylabel('Amplitudes');

% Add noise
noise_var = 0.01;
signals_n01 = addAudioSignalNoise(signals,noise_var);
noise_var = 0.0625;
signals_n025 = addAudioSignalNoise(signals,noise_var);

% See noisy signal
signal = signals_n01{1};
figure;
plot(signal);
title('Sinal Ruidoso - Comando Avançar');
xlabel('Amostras');
ylabel('Amplitudes');

% Build Samples with LPC Method
Tqmin = 0.015;      % Tamanho mínimo de um quadro: 15 ms
Nquad = 31;         % Numero de quadros
p = 10;             % Ordem do modelo AR(p)
Xlpc_n01 = linearPredictiveCoding(signals_n01,Fs,Tqmin,Nquad,p);
DATAlpc_n01.input = Xlpc_n01';
DATAlpc_n01.output = classes';
save('DATAlpc_n01.mat','DATAlpc_n01');
Xlpc_n025 = linearPredictiveCoding(signals_n025,Fs,Tqmin,Nquad,p);
DATAlpc_n025.input = Xlpc_n025';
DATAlpc_n025.output = classes';
save('DATAlpc_n025.mat','DATAlpc_n025');

% Build Samples with PSD Method
Xfreq_n01 = powerSpectralDensity(signals_n01,Fs);
DATApsd_n01.input = Xfreq_n01';
DATApsd_n01.output = classes';
save('DATApsd_n01.mat','DATApsd_n01');
Xfreq_n025 = powerSpectralDensity(signals_n025,Fs);
DATApsd_n025.input = Xfreq_n025';
DATApsd_n025.output = classes';
save('DATApsd_n025.mat','DATApsd_n025');

% Remove noise
L = 10;
signals_f01 = remAudioSignalNoise(signals_n01,L);
signals_f025 = remAudioSignalNoise(signals_n025,L);

% See filtered signal
signal = signals_f01{1};
figure;
plot(signal);
title('Sinal Ruidoso - Comando Avançar');
xlabel('Amostras');
ylabel('Amplitudes');

% Build Samples with LPC Method
Tqmin = 0.015;      % Tamanho mínimo de um quadro: 15 ms
Nquad = 31;         % Numero de quadros
p = 10;             % Ordem do modelo AR(p)
Xlpc_f01 = linearPredictiveCoding(signals_f01,Fs,Tqmin,Nquad,p);
DATAlpc_f01.input = Xlpc_f01';
DATAlpc_f01.output = classes';
save('DATAlpc_f01.mat','DATAlpc_f01');
Xlpc_f025 = linearPredictiveCoding(signals_f025,Fs,Tqmin,Nquad,p);
DATAlpc_f025.input = Xlpc_f025';
DATAlpc_f025.output = classes';
save('DATAlpc_f025.mat','DATAlpc_f025');

% Build Samples with PSD Method
Xfreq_f01 = powerSpectralDensity(signals_f01,Fs);
DATApsd_f01.input = Xfreq_f01';
DATApsd_f01.output = classes';
save('DATApsd_f01.mat','DATApsd_f01');
Xfreq_f025 = powerSpectralDensity(signals_f025,Fs);
DATApsd_f025.input = Xfreq_f025';
DATApsd_f025.output = classes';
save('DATApsd_f025.mat','DATApsd_f025');

%% (6.2) Adição de Ruído e Filtro PB - Compara desempenho

audioRecognition_pipeline;

%% (7.1) Geração de Novas Elocução de Voz

% Load Audio Data

sub_sampling = 2;
mean_rem = 1;
[signals,classes,~,Fs] = loadAudioDavid(sub_sampling,mean_rem);

% Build Samples with LPC Method

Tqmin = 0.015;      % Tamanho mínimo de um quadro: 15 ms
Nquad = 31;         % Numero de quadros
p = 10;             % Ordem do modelo AR(p)

Xlpc = linearPredictiveCoding(signals,Fs,Tqmin,Nquad,p);

DATAlpc_david.input = Xlpc';
DATAlpc_david.output = classes';
save('DATAlpc_david.mat','DATAlpc_david');

% Build Samples with PSD Method

Xfreq = powerSpectralDensity(signals,Fs);

DATApsd_david.input = Xfreq';
DATApsd_david.output = classes';
save('DATApsd_david.mat','DATApsd_david');

%% (7.2) Treina com elocuções originais, testa com novas elocuções

audioRecDavid_pipeline;

%% END


























