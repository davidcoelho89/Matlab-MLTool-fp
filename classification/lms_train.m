function [PARout] = lms_train(DATA,PAR)

% --- LMS classifier training ---
%
%   [PARout] = lms_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix                   [p x N]
%           output = labels matrix                      [Nc x N]
%       PAR.
%           Ne = maximum number of epochs               [cte]
%           eta = learning rate                         [0.01 0.1]
%           Von = enable or disable video               [cte]
%   Output:
%       PARout.
%           W = Regression / Classification Matrix      [Nc x p+1]
%           MQEtr = mean quantization error of training	[1 x Ne]
%           VID = frame structure (can be played with 'video function')

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.Ne = 200;       	% maximum number of training epochs
    PARaux.eta = 0.05;    	% Learning step
    PARaux.Von = 0;         % disable video 
    PAR = PARaux;
else
    if (~(isfield(PAR,'Ne')))
        PAR.Ne = 200;
    end
    if (~(isfield(PAR,'eta')))
        PAR.eta = 0.05;
    end
    if (~(isfield(PAR,'Von')))
        PAR.Von = 0;
    end
end

%% INITIALIZATIONS

% Data matrix
X = DATA.input;             % input matrix
Y = DATA.output;            % output matrix (desired values)

% Hyperparameters Init
Nep = PAR.Ne;              	% maximum number of epochs
eta = PAR.eta;              % learning rate
Von = PAR.Von;              % enable or disable video

% Problem Init
[Nc,~] = size(Y);           % Number of Classes (and neurons)
[p,N] = size(X);            % Number of attributes and samples
MQEtr = zeros(1,Nep);      	% Learning Curve

% Weight Matrix Init
if (isfield(PAR,'W')) 
    W = PAR.W;              % if already initialized
else
    W = 0.01*rand(Nc,p+1);  % else, initialize with small random numbers
end

% add bias to input matrix [x0 = +1]
X = [ones(1,N); X];          

% Initialize Video Structure
VID = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));

%% ALGORITHM

for ep = 1:Nep

    % Save frame of the current epoch
    if (Von)
        VID(ep) = hyperplane_frame(W,DATA);
    end
    
    % Shuffle Data
    I = randperm(N); 
    X = X(:,I); 
    Y = Y(:,I);   
    
    % Init sum of quadratic errors
    SQE = 0;
    
    for t = 1:N   % 1 epoch
        
        % Calculate Output
        x = X(:,t);             % Get sample vector
        U = W*x;                % Neurons Output vector
        Yh = U;                 % Don't have Activation Function
        e = Y(:,t) - Yh;    	% Error or each output
        W = W + eta*e*x';       % Update weights
        SQE = SQE + sum(e.^2);	% Sum of quadratic errors
        
    end % end of one epoch
    
    % Mean squared quantization error for each epoch
    MQEtr(ep) = SQE/N;

end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.MQEtr = MQEtr;
PARout.VID = VID;

%% END