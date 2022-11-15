function model_out = prototypesClassify(model,X)

% --- Prototype-Based Classify Function ---
%
%
%
%
%
%

%% SET DEFAULTO HYPERPARAMETERS

if(~isprop(model,'nearest_neighbors'))
    model.nearest_neighbors = 1;
end
if(~isprop(model,'knn_aproximation'))
    model.knn_aproximation = 1;
end

%% INITIALIZATIONS

% Get problem dimensions

[~,N] = size(X);                % Number of samples
[Nc,Nk] = size(model.Cy);       % Number of classes and prototypes

% Initialize outputs

y_h = -1*ones(Nc,N);            % One output for each sample
winners = zeros(1,N);        	% One closest prototype for each sample
distances = zeros(Nk,N);        % Distance from prototypes to each sample

if (K == 1)
    nearest_indexes = zeros(1,N);
else
    if (Nk <= K)
        nearest_indexes = zeros(Nk,N);
    else
        nearest_indexes = zeros(K+1,N);
    end
end

%% ALGORITHM



%% FILL OUTPUT STRUCTURE

model_out = model + X;

%% END