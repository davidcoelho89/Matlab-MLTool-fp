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
winners = zeros(1,N);       % The closest prototype for each sample
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
        d_min_all = -1;                     % Init global min distance
        
        for k = 1:Nk % for k = 1:Nk(1)
            
            prototype = model.Cx(:,k);      % Get prototype
            [~,class] = max(model.Cy(:,k));	% Get prototype label
            
            d = vectorsDistance(prototype,sample,model);
            distances(k,n) = d;
            
            % Get class-conditional closest prototype
            if(d_min(class) == -1 || d < d_min(class))
                d_min(class) = d;
            end

            % Get global closest prototype
            if(d_min_all == -1 || d < d_min_all)
                d_min_all = d;
                winners(n) = k;
                nearest_indexes(n) = k;
            end

        end

        % Fill Output
        for class = 1:Nc

            % Invert signal for 2nd class in binary problems

            if(class == 2 && Nc == 2)
            	Yh(2,:) = -Yh(1,:);
                break;
            end

            % Calculate Class Output for the Sample

            % Get minimum distance from class
            dp = d_min(class);
            % There is no prototypes from this class
            if (dp == -1)
                Yh(class,n) = -1;
            else
                % get minimum distance from other classes
                dm = -1;        
                for j = 1:Nc
                    if(j == class) % looking for other classes
                        continue;
                    elseif (d_min(j) == -1) % no prot from this class
                        continue;
                    elseif (dm == -1 || d_min(j) < dm)
                        dm = d_min(j);
                    end
                end
                if (dm == -1)  % no prototypes from other classes
                    Yh(class,n) = 1;
                else
                    Yh(class,n) = (dm - dp) / (dm + dp);
                end
            end

        end % end for class = 1:Nc
        
    end % end for n = 1:N
    
elseif (model.nearest_neighbors > 1) 
    
    for n = 1:N

        % Display classification iteration (for debug)
        if(mod(n,1000) == 0)
            disp(n);
        end

        % Get test sample
        sample = X(:,n);
        
        % Measure distance from sample to each prototype
        Vdist = zeros(1,Nk);
        for k = 1:Nk
            prototype = model.Cx(:,k);
            Vdist(k) = vectorsDistance(prototype,sample,model);
        end
        
        % hold distances
        distances(:,n) = Vdist';

        % Sort distances and get nearest neighbors
        sort_result = bubble_sort(Vdist,1);

        % Get closest prototype
        winners(n) = sort_result.ind(1);

        % Verify number of prototypes and neighbors
        if(Nk <= K)
            nearest_indexes(:,n) = sort_result.ind(1:Nk)';
            number_of_nearest = Nk;
        else
            nearest_indexes(:,n) = sort_result.ind(1:K+1)';
            number_of_nearest = K;
        end
        
        % Get labels of nearest neighbors
        lbls_near = model.Cy(:,nearest_indexes(:,n)');

        if(strcmp(model.knn_aproximation,'majority_voting'))
            
            % Compute votes
            votes = zeros(1,Nc);
            for k = 1:number_of_nearest
                [~,class] = max(lbls_near(:,k));
                votes(class) = votes(class) + 1;
            end
            
            % Update class
            [~,class] = max(votes);
            Yh(class,n) = 1;
        
        else % Weighted KNN

            % Avoid weights of 0
            epsilon = 0.001;

            % Auxiliary output and weight
            y_aux = zeros(Nc,1);
            w_sum = 0;
            
            % Get distances of nearest neighbors
            Dnear = Vdist(nearest_indexes(:,n)');
            
            % Calculate Output
            for k = 1:number_of_nearest
                % Compute Weight (Triangular)
                if (strcmp(model.knn_aproximation,'weighted_knn'))
                    Dnorm = Dnear(k)/(Dnear(end) + epsilon);
                    w = 1 - Dnorm;
                end
                w_sum = w_sum + w;
                % Compute weighted outptut
                y_aux = y_aux + w*lbls_near(:,k);

            end
            Yh(:,n) = y_aux / w_sum;

        end % end if majority_voting

    end % end for n = 1:N
    
end % end if(model.nearest_neighbors > 1)

%% FILL OUTPUT STRUCTURE

model_out = model;
model_out.Yh = Yh;
model_out.winners = winners;
model_out.distances = distances;
model_out.nearest_indexes = nearest_indexes;

end