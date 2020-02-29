function [Mred] = prototypes_select(DATA,PAR)

% --- Sample prototypes selection ---
%
%   [Mred] = prototypes_select(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%       PAR.
%           Ss = Selection strategy                             [cte]
%               1: Randomly
%               2: Renyi's entropy 
%               3: ALD
%               4: Coherence
%           v1 = Sparseness parameter 1                         [cte]
%           M = no of samples used to estimate kernel matrix    [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       Mred = reduced matrix with choosen samples [p x m]

%% INITIALIZATION

% Get data

X = DATA.input;     % Input matrix
[p,N] = size(X); 	% Total of attributes and samples

% Get hyperparameters

Ss = PAR.Ss;       	% prototype selection type
v1 = PAR.v1;       	% accuracy parameter (level of sparsity)
M = PAR.M;         	% number of samples used to estimate kernel matrix

%% ALGORITHM

% Shuffle data
I = randperm(N);
X = X(:,I);

% Randomly select from input
if (Ss == 1)
    
    % data is already shuffled
    Mred = X(:,1:M);

% Renyi's entropy method
elseif (Ss == 2),
    
    Mred = X(:,1:M);       	% get first m samples
    Mrem = X(:,M+1:end);	% remaining samples
    Nit = 100;              % Maximum number of iterations

    % initial entropy calculation
    Kmat = kernel_mat(Mred,PAR);
    entropy = ones(1,M)*Kmat*ones(M,1);
    
    % choose prototype samples
    for k = 1:Nit,
        % randomly select sample from reduced matrix
        I = randperm(M);
        red_s = Mred(:,I(1));
        % randomly select sample from remaining matrix
        J = randperm(N-M);
        rem_s = Mrem(:,J(1));
        % construct new reduced matrix
        Mred_new = Mred;
        Mred_new(:,I(1)) = rem_s;
        % construct new remaining matrix
        Mrem_new = Mrem;
        Mrem_new(:,J(1)) = red_s;
        % Calculate new entropy
        Kmat = kernel_mat(Mred_new,PAR);
        entropy_new = ones(1,M)*Kmat*ones(M,1);
        % replace old matrix for new ones
        if (entropy_new > entropy),
            entropy = entropy_new;
            Mred = Mred_new;
            Mrem = Mrem_new;
        end
    end
    
% Approximate Linear Dependency (ALD) Method    
elseif (Ss == 3),
    
    % init dictionary
    Dx = X(:,1);
    
    for t = 2:N,
        
        % get new sample
        xt = X(:,t);
        [~,M] = size(Dx);
        
        % calculate Kmat t-1
        Kmat = kernel_mat(Dx,PAR);
        
        % Calculate k t-1
        kt = zeros(M,1);
        for m = 1:M,
            kt(m) = kernel_func(Dx(:,m),xt,PAR);
        end
        
        % Calculate Ktt
        ktt = kernel_func(xt,xt,PAR);
        
        % Calculate coefficients
        at = Kmat\kt;
        
        % Calculate delta
        delta = ktt - kt'*at;
        
        % Expand or not dictionary
        if (delta > v1),
            Md_aux = zeros(p,M+1);
            Md_aux(:,1:M) = Dx;
            Md_aux(:,M+1) = xt;
            Dx = Md_aux;
        else
            
        end
    end
    Mred = Dx;

% Coherence Method    
elseif (Ss == 4),

    % init dictionary
    Dx = X(:,1);
    
    for t = 2:N,
        
        % get dictionary size
        [~,M] = size(Dx);
        
        % get new sample
        xt = X(:,t);
        
    	% Init coherence measure (first element of dictionary)
        u = kernel_func(Dx(:,1),xt,PAR) / ...
            (sqrt(kernel_func(Dx(:,1),Dx(:,1),PAR) * ...
            kernel_func(xt,xt,PAR)));
        u_max = abs(u);
        
        % get coherence measure
        if (M >= 2),
        for m = 2:M,
            % Calculate kernel
                u = kernel_func(Dx(:,m),xt,PAR) / ...
                    (sqrt(kernel_func(Dx(:,m),Dx(:,m),PAR) * ...
                    kernel_func(xt,xt,PAR)));
            % Calculate Coherence
            if (abs(u) > u_max),
                u_max = abs(u);
            end
        end
        end
        
        % Expand or not dictionary
        if(u_max <= v1)
            Dx_aux = zeros(p,M+1);
            Dx_aux(:,1:M) = Dx;
            Dx_aux(:,M+1) = xt;
            Dx = Dx_aux;
        end
        
    end

end

%% END