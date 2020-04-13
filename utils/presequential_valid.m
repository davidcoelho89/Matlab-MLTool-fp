function [PSout] = presequential_valid(DATA,HP,f_train,f_class)

% --- Presequential Validation Function ---
%
%   [PSout] = presequential_valid(DATA,HP,f_train,f_class)
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

X = DATA.input;                 % Attributes Matrix [pxN]
Y = DATA.output;             	% labels Matriz [cxN]

[~,N] = size(X);                % Number of samples

accuracy = 0;                   % Init accurary
Ds = 0;                         % Init # prototypes (dictionary size)

%% ALGORITHM

for t = 1:N,
    
    % ToDo - All
    
end

error = 1 - accuracy;

%% FILL OUTPUT STRUCTURE

PSout.err = error;
PSout.Ds = Ds;

%% END