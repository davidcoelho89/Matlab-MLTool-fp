function [DATAout] = hold_out(DATA,OPTIONS)

% --- Separates data between training and test ---
%
%   [DATAout] = hold_out(DATA,OPTIONS)
%
%   Input:
%       DATA.
%           input = input matrix                    [p x N]
%           output = output matrix                  [Nc x N] or [1 x N]
%           lbl = original labels                   [1 x N]
%       OPTIONS.
%           ptrn = % of data for training           [0 - 1]
%           hold = hold out method                  [cte]
%               1: all data randomly divided by tr (ptrn) and tst (1-ptrn) 
%               2: get class with the least amount of samples (nsamp). 
%                  calculate ntr (ptrn x nsamp).
%                  get ntr samples from each class for the training dataset
%                  the remaining samples are for the test dataset
%               3: separate samples of each class (nsamp_c).
%                  from each class, get "ptrn x nsamp_c" for training
%                  the remaining samples are for the test dataset
%   Output:
%       DATAout.
%           DATAtr = training samples
%           DATAts = test samples

%% INICIALIZAÇÕES

% Get Options
Mho = OPTIONS.hold;
ptrn = OPTIONS.ptrn;

% Get Data
X = DATA.input;
Y = DATA.output;
lbls = DATA.lbl;

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

switch (Mho)
    
    %------------- HOLD OUT -> ALEATORY CHOICE --------------%
    
    case(1)
        
        % Number of samples for training
        J = floor(ptrn*N);
        
        % Samples for training
        Xtr = X(:,1:J);
        Ytr = Y(:,1:J);
        lbl_tr = lbls(:,1:J);
        
        % Samples for test
        Xts = X(:,J+1:end);
        Yts = Y(:,J+1:end);
        lbl_ts = lbls(:,J+1:end);
        
	%------------- HOLD OUT -> BALANCED TRAINING ------------%
        
    case(2)
        
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
        J = floor(ptrn*Nmin);
        
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
        
	%------------- HOLD OUT -> SAME % PER CLASS -------------%

    case(3)
        
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
            J = floor(ptrn*n);
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
    
    %-------------------- OTHER OPTIONS ---------------------%

    otherwise
        
        disp('Type a correct option. Data not divided.')
        
end

%% FILL STRUCTURE

DATAout.DATAtr.input = Xtr;
DATAout.DATAtr.output = Ytr;
DATAout.DATAtr.lbl = lbl_tr;

DATAout.DATAts.input = Xts;
DATAout.DATAts.output = Yts;
DATAout.DATAts.lbl = lbl_ts;

%% END