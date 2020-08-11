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
%           Nh = number of hidden neurons  	                    [cte]
%           eta = learning rate           	                    [0.01 0.1]
%           mom = moment factor            	                    [0.5 1.0]
%           Nlin = non-linearity           	                    [cte]
%               1 -> Sigmoid                                    [0 e 1]
%               2 -> Hyperbolic Tangent                         [-1 e +1]
%           Von = enable or disable video                       [cte]
%   Output:
%       PARout.
%           W{1} = Weight Matrix: Inputs to Hidden Layer        [Nh x p+1]
%           W{2} = Weight Matrix: Hidden layer to output layer 	[No x Nh+1]
%           MQEtr = Mean Quantization Error (learning curve)	[Ne x 1]
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nh = 10;       	% Number of hidden neurons
    PARaux.Ne = 200;     	% Maximum training epochs
    PARaux.eta = 0.05;     	% Learning Step
    PARaux.mom = 0.75;    	% Moment Factor
    PARaux.Nlin = 2;       	% Non-linearity
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'Nh'))),
        PAR.Nh = 10;
    end
    if (~(isfield(PAR,'Ne'))),
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta'))),
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'mom'))),
        PAR.mom = 0.75;
    end
    if (~(isfield(PAR,'Nlin'))),
        PAR.Nlin = 2;
    end
    if (~(isfield(PAR,'Von'))),
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
[No,~] = size(D);           % Number of Classes and Output Neurons
[p,N] = size(X);            % attributes and samples
MQEtr = zeros(1,Nep);     	% Mean Quantization Error

% Weight Matrices Initialization

W = cell(2,1);                  % 1 hidden and 1 output layer
W_old = cell(2,1);              % 1 hidden and 1 output layer

if (isfield(PAR,'W')),
    W{1} = PAR.W{1};            % if already initialized
    W{2} = PAR.W{2};          	% if already initialized
else
    W{1} = 0.01*rand(Nh,p+1);	% Hidden Neurons weights
    W{2} = 0.01*rand(No,Nh+1);	% Output Neurons weights
end

W_old{1} = W{1};               	% necessary for moment factor
W_old{2} = W{2};             	% necessary for moment factor

% Add bias to input matrix
X = [ones(1,N);X];              % x0 = +1

% Initialize Video Structure
VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep,   % for each epoch

    % Update Parameters
    PAR.W = W;
    
    % Save frame of the current epoch
    if (Von),
        VID(ep) = get_frame_hyperplane(DATA,PAR,@mlp_classify);
    end
    
    % Shuffle Data
    I = randperm(N);        
    X = X(:,I);     
    D = D(:,I);   
    
    SQE = 0; % Init sum of quadratic errors
    
    for t = 1:N,   % for each sample
            
        % HIDDEN LAYER
        xi = X(:,t);              	% Get input sample
        Ui = W{1} * xi;            	% Activation of hidden neurons 
        Yi = mlp_f_ativ(Ui,Nlin);	% Non-linear function
        
        % OUTPUT LAYER
        xk = [+1; Yi];          	% build input of output layer
        Uk = W{2} * xk;           	% Activation of output neurons
        Yk = mlp_f_ativ(Uk,Nlin);	% Non-linear function
        
        % ERROR CALCULATION
        Ek = D(:,t) - Yk;           % error between real and estimated output
        SQE = SQE + sum(Ek.^2);     % sum of quadratic error
        
        % LOCAL GRADIENTS - OUTPUT LAYER
        Dk = mlp_f_gradlocal(Yk,Nlin);	% derivative of activation function
        DDk = Ek.*Dk;                  	% local gradient (output layer)
        
        % LOCAL GRADIENTS -  HIDDEN LAYER
        Di = mlp_f_gradlocal(Yi,Nlin);      % derivative of activation function
        DDi = Di.*(W{2}(:,2:end)'*DDk);     % local gradient (hidden layer)
        
        % WEIGHTS ADJUSTMENT - OUTPUT LAYER
        W_aux = W{2};                                    	% Hold curr weights
        W{2} = W{2} + eta*DDk*xk' + mom*(W{2} - W_old{2});	% Update curr weights
        W_old{2} = W_aux;                                	% Hold last weights
        
        % WEIGHTS ADJUSTMENT - HIDDEN LAYER
        W_aux = W{1};                                       % Hold curr weights
        W{1} = W{1} + eta*DDi*xi' + mom*(W{1} - W_old{1});	% Update curr weights
        W_old{1} = W_aux;                                   % Hold last weights
        
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