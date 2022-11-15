function model_out = prototypesClassify(model,X)

% --- Prototype-Based Classify Function ---
%
%
%
%
%
%

%% SET DEFAULT HYPERPARAMETERS

if(~isprop(model,'distance_measure'))
    model.distance_measure = 2;
end
if(~isprop(model,'nearest_neighbors'))
    model.nearest_neighbors = 1;
end
if(~isprop(model,'knn_aproximation'))
    model.knn_aproximation = 'majority_voting';
end
if(~isprop(model,'kernel_type'))
    model.kernel_type = 'none';
end
% if(~isprop(model,'regularization'))
%     model.regularization = 0.001;
% end

%% INITIALIZATIONS

% Get problem dimensions

[~,N] = size(X);        	% Number of samples
[Nc,Nk] = size(model.Cy);	% Number of classes and prototypes

% Initialize outputs

Yh = -1*ones(Nc,N);         % One output for each sample
winners = zeros(1,N);       % One closest prototype for each sample
distances = zeros(Nk,N);	% Distance from prototypes to each sample

% indexes for nearest prototypes
if (model.nearest_neighbors == 1)
    nearest_indexes = zeros(1,N);
else
    if (Nk <= model.nearest_neighbors)
        nearest_indexes = zeros(Nk,N);
    else
        nearest_indexes = zeros(model.nearest_neighbors+1,N);
    end
end

%% ALGORITHM

if (model.nearest_neighbors == 1)
    
    for n = 1:N

        % Display classification iteration (for debug)
        if(mod(n,1000) == 0)
            disp(n);
        end
        
        % Get test sample
        sample = X(:,n);
        
        % Get closest prototype and min distance from sample to each class
        d_min = -1*ones(Nc,1);              % Init class min distance
        d_min_all = -1;                     % Initi global min distance
        % for k = 1:Nk(1)
        for k = 1:Nk
            prot = model.Cx(:,k);           % Get prototype
            [~,class] = max(model.Cy(:,k));	% Get prototype label
            
        end
        
        
    end
    
elseif (model.nearest_neighbors > 1) 
    
    for n = 1:N
        
    end
    
end

%% FILL OUTPUT STRUCTURE

model_out = model + X;

%% END













