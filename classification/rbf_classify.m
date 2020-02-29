function [OUT] = rbf_classify(DATA,PAR)

% --- RBF Classifier Test ---
%
%   [OUT] = rbf_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                  [p x N]
%       PAR.
%           ativ = activation function type     [cte]
%           q = Number of hidden neurons        [cte]
%           r = radius (spread) of basis        [Nh x 1]
%           W = Hidden layer weight Matrix      [Nh x p]
%           M = Output layer weight Matrix      [No x Nh+1]
%   Output:
%       PARout.
%           y_h = classifier's output        	[Nc x N]

%% INITIALIZATION

% Get data
X = DATA.input;             % Get attribute matrix

% Get parameters
ativ = PAR.ativ;            % activation function type  
Nh = PAR.Nh;               	% Number of hidden neurons
W = PAR.W;                  % hidden layer weight Matrix
r = PAR.r;                  % radius of each rbf
M = PAR.M;                  % Output layer weight Matrix

% Problem Init
[~,N] = size(X);            % Number of samples
[Nc,~] = size(M);           % Number for classes / outputs

% Initialize Outputs
y_h = zeros(Nc,N);          % Estimated output

%% ALGORITHM

for t = 1:N,
    
    % Get input sample
    xi = X(:,t);
 
    % Calculate output of basis functions
    z_vec = zeros(Nh,1);          	% init output of basis functions
    for i = 1:Nh,       
        ci = W(:,i);                % Get rbf center
        ui = sum((xi - ci).^2);    	% Calculates distance
        ri = r(i);                  % get radius
        z_vec(i) = rbf_f_ativ(ui,ri,ativ);
    end
    z_vec = z_vec/sum(z_vec);       % normalize hidden layer outputs
    z1 = [1;z_vec];                 % add bias to hidden layer outputs
   
    % Calculate output of neural net
    y_h(:,t) = M*z1;
    
end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END