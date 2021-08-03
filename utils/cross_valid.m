function [CVout] = cross_valid(DATA,HP,f_train,f_class,CVp)

% --- Cross Validation Function ---
%
%   [CVout] = cross_valid(DATA,HP,f_train,f_class,CVp)
%
%   Input:
%       DATA.
%           input = Matrix of training attributes             	[p x N]
%           output = Matrix of training labels                 	[Nc x N]
%       HP = set of HyperParameters to be tested
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%       CVp.
%           fold = number of data partitions                        [cte]
%           type = type of cross validation                         [cte]
%               1: takes into account just accurary
%               2: takes into account also the dicitionary size
%           lambda = trade-off between error and dictionary size   	[cte]
%   Output:
%       CVout.
%           metric = metric to be minimized
%           acc = mean accuracy for data set and parameters
%           err = mean error for data set and parameters
%           Ds = percentage of prototypes compared to the dataset

%% INIT

% Get Data 

X = DATA.input;     	% Attributes Matrix [p x N]
Y = DATA.output;     	% labels Matriz [Nc x N]
[~,N] = size(X);      	% Number of samples

% Get HyperParameters

if (nargin == 4)
    CVp.fold = 5;       % 5-fold cross-validation
    CVp.type = 1;       % just takes into account accuracy
    CVp.lambda = 0.5;   % More weight for dictionary size
end

Nfold = CVp.fold;     	% Number of data partitions
part = floor(N/Nfold); 	% Size of each data partition
type = CVp.type;        % If classifier is prototype-based or not
lambda = CVp.lambda;    % trade-off between error and dictionary size

% Init Outupts

accuracy = 0;          	% Init accurary
Ds = 0;                 % Dictionary Size

%% ALGORITHM

for fold = 1:Nfold

    % Define Data division

    if fold == 1
        DATAtr.input  = X(:,part+1:end);
        DATAtr.output = Y(:,part+1:end);
        DATAts.input  = X(:,1:part);
        DATAts.output = Y(:,1:part);
    elseif fold == Nfold
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
    [PAR] = f_train(DATAtr,HP);
    
    % Accumulate Number of Prototypes (for prototype-based classifiers)
    if (type == 2)
        [~,Nk] = size(PAR.Cx);
        Ds = Ds + Nk;
    end

    % Test of classifier
    [OUT] = f_class(DATAts,PAR);

    % Statistics of Classifier
    [STATS_ts] = class_stats_1turn(DATAts,OUT);

    % Accumulate Accuracy rate
    accuracy = accuracy + STATS_ts.acc;

end

accuracy = accuracy / Nfold;    % Mean Accuracy
error = 1 - accuracy;           % Mean Error
Ds = Ds / (N * Nfold);          % Mean Percentage of Prototypes

if (type == 1)
    metric = error;
elseif (type == 2)
    metric = Ds + lambda * error;
end

%% FILL OUTPUT STRUCTURE

CVout.metric = metric;
CVout.acc = accuracy;
CVout.err = error;
CVout.Ds = Ds;

%% END