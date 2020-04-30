function [CVout] = cross_valid2(DATA,HP,f_train,f_class,CVp)

% --- Cross Validation Function ---
%
%   [CVout] = cross_valid2(DATA,HP,f_train,f_class,CVp)
%
%   Input:
%       DATA.
%           input = Matrix of training attributes             	[p x N]
%           output = Matrix of training labels                 	[Nc x N]
%       HP = set of HyperParameters to be tested
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%       CVp.
%           Nfold = number of partitions                       	[cte]
%   Output:
%       CVout.
%           err = mean error for data set and parameters
%           np = percentage of prototypes compared to the dataset

%% SET DEFAULT HYPERPARAMETERS



%% INIT

X = DATA.input;                 % Attributes Matrix [pxN]
Y = DATA.output;             	% labels Matriz [cxN]

[~,N] = size(X);                % Number of samples

Nfold = CVp.fold;               % Number of folds
part = floor(N/Nfold);          % Size of each data partition

accuracy = 0;                   % Init accurary
Ds = 0;                         % Init # prototypes (dictionary size)

%% ALGORITHM

for fold = 1:Nfold;

    % Define Data division

    if fold == 1,
        DATAtr.input  = X(:,part+1:end);
        DATAtr.output = Y(:,part+1:end);
        DATAts.input  = X(:,1:part);
        DATAts.output = Y(:,1:part);
    elseif fold == Nfold,
        DATAtr.input  = X(:,1:(Nfold-1)*part);
        DATAtr.output = Y(:,1:(Nfold-1)*part);
        DATAts.input  = X(:,(Nfold-1)*part+1:end);
        DATAts.output = Y(:,(Nfold-1)*part+1:end);
    else
        DATAtr.input  = [X(:,1:(fold-1)*part) X(:,fold*part+1:end)];
        DATAtr.output = [Y(:,1:(fold-1)*part) Y(:,fold*part+1:end)];
        DATAts.input  = X(:,(fold-1)*part+1:fold*part);
        DATAts.output = Y(:,(fold-1)*part+1:fold*part);
    end

    % Training of classifier
    [PARo] = f_train(DATAtr,HP);

    % Acc Number of Prototypes
    size_Cx = size(PARo.Cx);
    Ds_fold = prod(size_Cx(2:end));
    Ds = Ds + Ds_fold;

    % Dont measure accuracy if number of prototypes is too high
    if (Ds_fold > 400)
        continue;
    end

    % Test of classifier
    [OUT] = f_class(DATAts,PARo);

    % Statistics of Classifier
    [STATS_ts] = class_stats_1turn(DATAts,OUT);

    % Get Accuracy rate
    accuracy = accuracy + STATS_ts.acc;

end

accuracy = accuracy / Nfold;	% Mean Accuracy
error = 1 - accuracy;           % Mean Error
Ds = (Ds)/(N * Nfold);       	% Mean number of prototypes

%% FILL OUTPUT STRUCTURE

CVout.err = error;
CVout.Ds = Ds;

%% END