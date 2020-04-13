function [PSout] = presequential_valid1(DATA,HP,f_train,f_class)

% --- Presequential Validation Function ---
%
%   [PSout] = presequential_valid1(DATA,HP,f_train,f_class)
%
%   Input:
%       DATA.
%           input = Matrix of training attributes             	[p x N]
%           output = Matrix of training labels                 	[Nc x N]
%       HP = set of HyperParameters to be tested
%       f_train = handler for classifier's training function
%       f_class = handler for classifier's classification function       
%   Output:
%       CVout.
%           err = mean error for data set and parameters
%           np = percentage of prototypes compared to the dataset

%% INIT

[~,N] = size(DATA.input);       % Number of samples

accuracy = 0;                   % Init accurary
Ds = 0;                         % Init # prototypes (dictionary size)

%% ALGORITHM

% Add first element to dictionary

DATAn.input = DATA.input(:,1);      % First element input
DATAn.output = DATA.output(:,1);    % First element output
PAR = k2nn_train(DATAn,HP);         % Add element

for n = 2:N,
    
    % Get current data
    
    DATAn.input = DATA.input(:,n);
    DATAn.output = DATA.output(:,n);
    
	[~,max_y] = max(DATAn.output);
    
    % Test (classify arriving data with current model)
    
	OUTn = f_class(DATAn,PAR);
	[~,max_yh] = max(OUTn.y_h);
    
    % Statistics
    if(max_y == max_yh),
        accuracy = accuracy + 1;
    end
    
    % Update score (for prunning method)
    
    PAR = k2nn_score_updt(DATAn,PAR,OUTn);
    
    % Train (with arriving data)
    
    PAR = f_train(DATAn,PAR);
    
end

% Get error

accuracy = accuracy / N;
error = 1 - accuracy;

% Get Dictionary Size

[~,Nk] = size(PAR.Cx);
Ds = Ds + Nk/N;

%% FILL OUTPUT STRUCTURE

PSout.err = error;
PSout.Ds = Ds;

%% END