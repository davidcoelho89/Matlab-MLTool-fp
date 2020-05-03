function [PSout] = presequential_valid2(DATA,HP,f_train,GSp)

% --- Presequential Validation Function ---
%
%   [PSout] = presequential_valid2(DATA,HP,f_train,f_class)
%
%   Input:
%       DATA.
%           input = Matrix of training attributes             	[p x N]
%           output = Matrix of training labels                 	[Nc x N]
%       HP = set of HyperParameters to be tested
%       f_train = handler for classifier's training function
%       GSp.
%           lambda = trade-off between error and dictionary size [0 - 1]
%           preseq_type = type of presenquential validation
%               1: k2nn 
%               2: isk2nn      
%   Output:
%       CVout.
%           PAR = hold parameters of dictionary                 [struct]
%           metric = value to be minimized                      [cte]

%% INIT

[~,N] = size(DATA.input);       % Number of samples
[Nc,~] = size(DATA.output);     % Number of classes

lambda = GSp.lambda;            % trade-off between error and dict size

max_prot = HP.max_prot;         % max number of prototypes

accuracy = 0;                   % Init accurary
Ds = 0;                         % Init # prototypes (dictionary size)

%% ALGORITHM

% Add first element to dictionary

DATAn.input = DATA.input(:,1);      % First element input
DATAn.output = DATA.output(:,1);    % First element output
PAR = f_train(DATAn,HP);            % Add element

for n = 2:N,
    
    % Get current data
    
    DATAn.input = DATA.input(:,n);
    DATAn.output = DATA.output(:,n);
	[~,max_y] = max(DATAn.output);
    
    % Test (classify arriving data with current model)
    % Train (update model with arriving data)
    
    PAR = f_train(DATAn,PAR);
    [~,max_yh] = max(PAR.y_h);
    
    % Statistics

    if(max_y == max_yh),
        accuracy = accuracy + 1;
    end
    
    % Restriction: Max number of prototypes
    [~,Nk] = size(PAR.Cx);
    if (Nk >= PAR.max_prot),
        break;
    end
    
end

% Get error

accuracy = accuracy / N;
error = 1 - accuracy;

% Get Dictionary Size

[~,Nk] = size(PAR.Cx);
Ds = Ds + Nk/N;

% Generate Metric (value to be minimized)

if (Nk <= Nc || Nk > max_prot),
    metric = 1 + lambda;
else
    metric = Ds + lambda * error;
end

%% FILL OUTPUT STRUCTURE

PSout.PAR = PAR;
PSout.metric = metric;

%% END