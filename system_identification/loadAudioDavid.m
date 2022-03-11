function [X,y,sig_mean,fs_sub] = loadAudioDavid(subsampling,mean_rem)

% --- Build Signals cells and its Labels from audio files ---
%
%   [X,y,sig_mean,fs_sub] = loadAudioDavid(subsampling,mean_rem)
%
%   Input:
%       subsampling = generate more samples from each signal    [cte]
%       mean_rem = remove (or not) mean of each signal          [cte]
%   Output:
%       X = Cell with all signals                               [Ns x 1]
%       y = Vector of signals' lables                           [Ns x 1]
%       sig_mean = mean of each signal                          [Ns x 1]
%       fs_sub = frequency of sub sampled signals               [cte]

%% INIT

comandos = {"avancar_","direita_","esquerda_","parar_","recuar_"};

Ncom = length(comandos);
Nrep = 2;
mono_or_stereo = 2;
sub = subsampling;
Nsamples = Ncom * Nrep * sub * mono_or_stereo;

X = cell(Nsamples,1);
y = zeros(Nsamples,1);
sig_mean = zeros(Nsamples,1);

cont = 0;
str1 = "comando_";
% str2 = "which comand"
% str3 = "number of command repetition"
str4 = ".wav";

%% ALGORITHM

for i = 1:Ncom
    str2 = comandos{i};
    for j = 1:Nrep
        str3 = int2str(j+10);
        % Build filename
        filename = strcat(str1,str2,str3,str4);
        % Load signals
        [Y,Fs] = audioread(filename);
        % disp(Fs); % for debug
        Y = Y';
        [Nsig,len] = size(Y);
        % Build Inputs (cell) and Labels (vector)
        for k = 1:sub
            n = k:sub:len;
            Ysub = Y(:,n);
            for sig = 1:Nsig
                cont = cont + 1;
                y(cont) = i;
                X{cont} = Ysub(sig,:);
            end
            
        end
        
    end
end

for i = 1:Nsamples
    signal = X{i};
    sig_mean(i) = mean(signal);
    if(mean_rem)
        X{i} = X{i} - sig_mean(i);
    end
end

fs_sub = floor(Fs/subsampling);

%% END