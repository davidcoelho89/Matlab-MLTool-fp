function [PARout] = mlp_train(DATA,PAR)

% --- MLP Classifier Training ---
%
%   [PARout] = mlp_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                                  [p x N]
%           output = labels                                     [No x N]
%       PAR.
%           Ne = maximum number of epochs	                    [cte]
%           Nh = number of hidden neurons                       [NL-1 x 1]
%           eta = learning rate           	                    [0.01 0.1]
%           mom = moment factor            	                    [0.5 1.0]
%           Nlin = non-linearity           	                    [cte]
%               1 -> Sigmoid                                    [0 e 1]
%               2 -> Hyperbolic Tangent                         [-1 e +1]
%           Von = enable or disable video                       [cte]
%   Output:
%       PARout.
%           W = weight matrices                                 [NL x 1]
%               W{1:end-1} = Hidden layer weight Matrix 
%               W{end} = Output layer weight Matrix
%           MQEtr = Mean Quantization Error (learning curve)	[Ne x 1]
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.Nh = 10;       	% Number of hidden neurons
    PARaux.Ne = 200;     	% Maximum training epochs
    PARaux.eta = 0.05;     	% Learning Step
    PARaux.mom = 0.75;    	% Moment Factor
    PARaux.Nlin = 2;       	% Non-linearity
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'Nh')))
        PAR.Nh = 10;
    end
    if (~(isfield(PAR,'Ne')))
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta')))
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'mom')))
        PAR.mom = 0.75;
    end
    if (~(isfield(PAR,'Nlin')))
        PAR.Nlin = 2;
    end
    if (~(isfield(PAR,'Von')))
        PAR.Von = 0;
    end
end

%% INITIALIZATIONS

% Data Initialization
X = DATA.input;             % Input Matrix
D = DATA.output;            % Output Matrix

% Hyperparameters Initialization
Nep = PAR.Ne;           	% Number of training epochs
Nh = PAR.Nh;                % Number of hidden neurons
eta = PAR.eta;              % learning rate 
mom = PAR.mom;              % Moment Factor
Nlin = PAR.Nlin;            % Non-linearity
Von = PAR.Von;              % Enable or disable video

% Problem Initialization
[Nc,~] = size(D);           % Number of Classes and Output Neurons
[p,N] = size(X);            % attributes and samples
MQEtr = zeros(1,Nep);     	% Mean Quantization Error
structure = [p,Nh,Nc];      % MLP Structure
NL = length(structure) - 1; % Number of layers

disp('MLP Structure:');
disp(structure);

% Cells for holding info of forward and backward steps
x = cell(NL,1);             % input of each layer
y = cell(NL,1);             % output of each layer
delta = cell(NL,1);         % local gradient of each layer

% Weight Matrices Initialization (NL-1 Hidden layers)

W = cell(NL,1);             % Weight Matrices
W_old = cell(NL,1);         % Necessary for moment factor

if (isfield(PAR,'W'))       % if already initialized
    for i = 1:NL
        W{i} = PAR.W{i};
    end
else                        % Initialize randomly
    for i = 1:NL
        W{i} = 0.01*rand(structure(i+1),structure(i)+1);
        W_old{i} = W{i};    
    end
end

% Add bias to input matrix
X = [ones(1,N) ; X];       % x0 = +1

% Initialize Video Structure
VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep   % for each epoch

    % Update Parameters
    PAR.W = W;
    
    % Save frame of the current epoch
    if (Von)
        VID(ep) = get_frame_hyperplane(DATA,PAR,@mlp_classify);
    end
    
    % Shuffle Data
    I = randperm(N);        
    X = X(:,I);     
    D = D(:,I);   
    
    SQE = 0; % Init sum of quadratic errors
    
    for t = 1:N   % for each sample
        
        % Forward Step (Calculate Layers' Outputs)
        for i = 1:NL
            if (i == 1) % Input layer
                x{i} = X(:,t);          % Get input data
            else
                x{i} = [+1; y{i-1}];    % add bias to last output
            end
            Ui = W{i} * x{i};           % Activation of hidden neurons
            y{i} = mlp_f_ativ(Ui,Nlin); % Non-linear function
        end
        
        % Error Calculation
        E = D(:,t) - y{NL};
        SQE = SQE + sum(E.^2);
        
        % Backward Step (Calculate Layers' Local Gradients)
        for i = NL:-1:1
            f_der = mlp_f_gradlocal(y{i},Nlin);
            if (i == NL) % output layer
                delta{i} = E.*f_der;
            else
                delta{i} = f_der.*(W{i+1}(:,2:end)'*delta{i+1});
            end
        end
        
        % Weight Adjustment
        for i = NL:-1:1
            W_aux = W{i};
            W{i} = W{i} + eta*delta{i}*x{i}' + mom*(W{i} - W_old{i});
            W_old{i} = W_aux;
        end

    end   % end of epoch
        
    % Mean Squared Error of epoch
    MQEtr(ep) = SQE/N;
    
end   % end of all epochs

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.MQEtr = MQEtr;
PARout.VID = VID;

%% END