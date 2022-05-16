function [datasets] = splitDataset(dataset_in,split_method,percentage_for_training)

% --- Separates data between training and test ---
%
%   [datasets] = splitDataset(DATA,OPTIONS)
%
%   Input:
%       dataset_in.
%           input = input matrix                    [p x N]
%           output = output matrix                  [Nc x N] or [1 x N]
%           lbl = original labels                   [1 x N]
%    	hold = hold out method                      [cte]
%       	'random': 
%             - All data randomly divided by training (ptrn) and test (1 - ptrn) 
%       	'balanced_training': 
%             - Get class with the least amount of samples (nsamp). 
%             - Calculate ntr (ptrn x nsamp).
%             - Get ntr samples from each class for the training dataset
%             - The remaining samples are for the test dataset
%           'class_distribution': 
%             - Separate samples of each class (nsamp_c).
%             - From each class, get "ptrn x nsamp_c" for training
%             - The remaining samples are for the test dataset
%       percentage_for_training                     [0 - 1]
%   Output:
%       datasets.
%           data_tr = training samples
%           data_ts = test samples

%% INITIALIZATIONS

% Get Data
X = dataset_in.input;
Y = dataset_in.output;
lbls = dataset_in.lbl;

% Number of classes and samples
[Nc,N] = size(Y);
flag_seq = 0;
if Nc == 1
    flag_seq = 1;           % Informs that labels are sequential
    Nc = length(unique(Y)); % calculates one more time the number of classes
end

% Shuffle Data
I = randperm(N);
X = X(:,I);
Y = Y(:,I);
lbls = lbls(:,I);

% Init Outputs
Xtr = [];
Ytr = [];
lbl_tr = [];
Xts = [];
Yts = [];
lbl_ts = [];

%% ALGORITMO

if(strcmp(split_method,'random'))

    % Number of samples for training
    J = floor(percentage_for_training*N);

    % Samples for training
    Xtr = X(:,1:J);
    Ytr = Y(:,1:J);
    lbl_tr = lbls(:,1:J);

    % Samples for test
    Xts = X(:,J+1:end);
    Yts = Y(:,J+1:end);
    lbl_ts = lbls(:,J+1:end);

elseif(strcmp(split_method,'balanced_training'))

    % Initialize auxiliary variables

    X_c = cell(Nc,1);
    Y_c = cell(Nc,1);
    lbl_c = cell(Nc,1);
    for i = 1:Nc
        X_c{i} = [];
        Y_c{i} = [];
        lbl_c{i} = [];
    end

    % Separate data of each class

    for i = 1:N
        % current sample
        xi = X(:,i);
        yi = Y(:,i);
        lbl_i = lbls(:,i);
        % define class
        if (flag_seq == 1)
            class = yi;
        else
            class = find(yi > 0);
        end
        % adiciona amostra à matriz correspondente
        X_c{class} = [X_c{class} xi];
        Y_c{class} = [Y_c{class} yi];
        lbl_c{class} = [lbl_c{class} lbl_i];
    end

    % Get minimum quantity of samples at one class
    
    for i = 1:Nc
        if (i == 1)
            % init min number of samples
            [~,Nmin] =  size(X_c{i});
        else
            % verify the smaller number
            [~,n] =  size(X_c{i});
            if (n < Nmin)
                Nmin = n;
            end
        end
    end

    % Quantidade de amostras, para treinamento, de cada classe
    
    J = floor(percentage_for_training*Nmin);

    for i = 1:Nc
        % Number of samples from class i
        [~,n] =  size(X_c{i});
        % Shuffle samples from class i
        I = randperm(n);
        % Inputs for training and test
        Xtr = [Xtr X_c{i}(:,I(1:J))];
        Xts = [Xts X_c{i}(:,I(J+1:end))];
        % Outputs for training and test
        Ytr = [Ytr Y_c{i}(:,I(1:J))];
        Yts = [Yts Y_c{i}(:,I(J+1:end))];
        % Labels for training and test
        lbl_tr = [lbl_tr lbl_c{i}(:,I(1:J))];
        lbl_ts = [lbl_ts lbl_c{i}(:,I(J+1:end))];
    end

elseif(strcmp(split_method,'class_distribution'))
    
    % Initialize auxiliary variables

    X_c = cell(Nc,1);
    Y_c = cell(Nc,1);
    lbl_c = cell(Nc,1);
    for i = 1:Nc
        X_c{i} = [];
        Y_c{i} = [];
        lbl_c{i} = [];
    end

    % Separate data of each class

    for i = 1:N
        % current sample
        xi = X(:,i);
        yi = Y(:,i);
        lbl_i = lbls(:,i);
        % define class
        if (flag_seq == 1)
            class = yi;
        else
            class = find(yi > 0);
        end
        % adiciona amostra à matriz correspondente
        X_c{class} = [X_c{class} xi];
        Y_c{class} = [Y_c{class} yi];
        lbl_c{class} = [lbl_c{class} lbl_i];
    end

    % Get % ptrn of each samples form class

    for i = 1:Nc
        % Number of samples from class i
        [~,n] =  size(X_c{i});
        % Shuffle samples from class i
        I = randperm(n);
        % Number of samples, from class i, for training
        J = floor(percentage_for_training*n);
        % Inputs for training and test
        Xtr = [Xtr X_c{i}(:,I(1:J))];
        Xts = [Xts X_c{i}(:,I(J+1:end))];
        % Outputs for training and test
        Ytr = [Ytr Y_c{i}(:,I(1:J))];
        Yts = [Yts Y_c{i}(:,I(J+1:end))];
        % Labels for training and test
        lbl_tr = [lbl_tr lbl_c{i}(:,I(1:J))];
        lbl_ts = [lbl_ts lbl_c{i}(:,I(J+1:end))];
    end
    
else
    disp('Type a correct option. Data not divided.')
end

%% FILL STRUCTURE

datasets.data_tr.input = Xtr;
datasets.data_tr.output = Ytr;
datasets.data_tr.lbl = lbl_tr;

datasets.data_ts.input = Xts;
datasets.data_ts.output = Yts;
datasets.data_ts.lbl = lbl_ts;

%% END