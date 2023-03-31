classdef prototypeBasedClassifier
    %
    % --- Common features for all Prototype-Based Classifiers ---
    %
    % Properties (Hyperparameters)
    %  
    %   - distance_measure = which measure used to compare two vectors
    %      (used in a lot of functions for prototype-based classifiers)
    %   - nearest_neighbors = number of nearest neighbors (classification)
    %   - knn_aproximation = how the output will be generated
    %       = 'majority_voting'
    %       = 'weighted_knn'
    %   - kernel_type = which kernel used (kernel based classifiers)
    %       = 'none' (it is not a kernel based model)
    %
    % Properties (Parameters)
    %
    %   - Cx = Clusters' centroids (prototypes)
    %   - Cy = Clusters' labels
    %   - Yh = all predictions (predict function) [Nc x N]
    %	- winners = The closest prototype for each sample [1 x N]
    %	- distances = Distance from prototypes to each sample [Nk x N]
    %	- nearest_indexes = identify nearest prototypes for each sample [K x N]
    %
    % Methods
    %
    %   - prototypeBasedClassifier()    % Constructor
    %   - predict(self,X)               % Prediction Function
    %
    % ----------------------------------------------------------------

    % Hyperparameters
    properties
        distance_measure = 2;
        nearest_neighbors = 1;
        knn_aproximation = 'majority_voting';
        kernel_type = 'none';
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Cx = [];              % Clusters' centroids (prototypes)
        Cy = [];              % Clusters' labels
        Yh = [];	          % All predictions (predict function)
        yh = [];              % Last prediction (partial_predict function)
        winners = [];         % Closest prototypes to each sample  [1 x N]
        winner = [];          % Closest prototypes to a sample     [1 x 1]
        distances = [];       % Distance of prot to each sample    [Nk x N]
        distance = [];        % Distance of prot to a sample       [Nk x 1]
        nearest_indexes = []; % Nearest prototypes for each sample [K x N]
        nearest_index = [];   % Nearest prototypes for a sample    [K x 1]

    end

    methods

        % Constructor
        function self = prototypeBasedClassifier()
            % Set the hyperparameters after initializing!
        end

        % Training Function (1 instance)
        % function self = partial_fit(self,x,y)
        %     % Each classifier has it own function
        % end

        % Training Function (N instances)
        % function self = fit(self,X,Y)
        %   "Each classifier has it own function"
        % end
        
        % Prediction Function (1 instance)
        function self = partial_predict(self,x)
            
            % Number of classes and prototypes
            [Nc,Nk] = size(self.Cy);
            
            % Init variables
            self.distance = zeros(Nk,1);
            self.yh = -1*ones(Nc,1);
            
            % 1 Nearest Neighbor Case
            if (self.nearest_neighbors == 1)
                
                % Get closest prototype and min distance 
                % from sample to each class

                d_min = -1*ones(Nc,1);  % Init class min distance
                d_min_all = -1;         % Init global min distance
                
                for k = 1:Nk
                    
                    prototype = self.Cx(:,k);
                    [~,class] = max(self.Cy(:,k));
                    
                    d = vectorsDistance(prototype,x,self);
                    self.distance(k,1) = d;
                    
                    % Get class-conditional closest prototype
                    if(d_min(class) == -1 || d < d_min(class))
                        d_min(class) = d;
                    end
                    
                    % Get global closest prototype
                    if(d_min_all == -1 || d < d_min_all)
                        d_min_all = d;
                        self.winner = k;
                        self.nearest_index = k;
                    end
                    
                end
                    
                % Fill Output
                for class = 1:Nc

                    % Invert signal for 2nd class in binary problems
                    if(class == 2 && Nc == 2)
                        self.yh(2) = -self.yh(1);
                        break;
                    end

                    % Calculate Class Output for the Sample

                    % Get minimum distance from class
                    dp = d_min(class);
                    % There is no prototypes from this class
                    if (dp == -1)
                        self.yh(class) = -1;
                    else
                        % get minimum distance from other classes
                        dm = -1;        
                        for j = 1:Nc
                            % looking for other classes
                            if(j == class)
                                continue;
                            % no prot from this class
                            elseif (d_min(j) == -1)                                
                                continue;
                            % get distance
                            elseif (dm == -1 || d_min(j) < dm)
                                dm = d_min(j);
                            end
                        end
                        % no prototypes from other classes
                        if (dm == -1)  
                            self.yh(class) = 1;
                        else
                            self.yh(class) = (dm - dp) / (dm + dp);
                        end
                    end

                end % end fill output (for class = 1:Nc)
                
            % K Nearest Neighbors Case
            elseif(self.nearest_neighbors > 1)    
                
                % Init Nearest Index
                if (Nk <= self.nearest_neighbors)
                    self.nearest_index = zeros(Nk,1);
                else
                    self.nearest_index = zeros(self.nearest_neighbors+1,1);
                end
                
                % Measure distance from sample to each prototype
                Vdist = zeros(Nk,1);
                for k = 1:Nk
                    prototype = self.Cx(:,k);
                    Vdist(k) = vectorsDistance(prototype,x,self);
                end
                
                % Hold distances
                self.distance = Vdist;
                
                % Sort distances and get nearest neighbors
                sort_result = bubble_sort(Vdist,1);
                
                % Get closest prototype
              	self.winner = sort_result.ind(1);
                
                % Verify number of prototypes and neighbors
                K = self.nearest_neighbors;
                if(Nk <= K)
                    self.nearest_index = sort_result.ind(1:Nk)';
                    number_of_nearest = Nk;
                else
                    self.nearest_index = sort_result.ind(1:K+1)';
                    number_of_nearest = K;
                end
                
                % Get labels of nearest neighbors
                lbls_near = self.Cy(:,self.nearest_index');
                
                if(strcmp(self.knn_aproximation,'majority_voting'))
                    
                    % Compute votes
                    votes = zeros(1,Nc);
                    for k = 1:number_of_nearest
                        [~,class] = max(lbls_near(:,k));
                        votes(class) = votes(class) + 1;
                    end
                    
                    % Update class
                    [~,class] = max(votes);
                    self.yh(class) = 1;
                    
                else % Weighted KNN
                    
                    % Avoid weights of 0
                    epsilon = 0.001;
                    
                    % Auxiliary output and weight
                    y_aux = zeros(Nc,1);
                    w_sum = 0;
                    
                    % Get distances of nearest neighbors
                    Dnear = Vdist(self.nearest_index');
                    
                    % Calculate Output
                    for k = 1:number_of_nearest
                        % Compute Weight (Triangular)
                        if (strcmp(self.knn_aproximation,'weighted_knn'))
                            Dnorm = Dnear(k)/(Dnear(end) + epsilon);
                            w = 1 - Dnorm;
                        end
                        w_sum = w_sum + w;
                        % Compute weighted outptut
                        y_aux = y_aux + w*lbls_near(:,k);

                    end
                    self.yh = y_aux / w_sum;
                    
                end % end strcmp(self.knn_aproximation,'majority_voting')

            end % end self.nearest_neighbors == 1
            
        end

        % Prediction Function (N instances)
        function self = predict(self,X)

            % Get problem dimensions

            [~,N] = size(X);        	% Number of samples
            [Nc,Nk] = size(self.Cy);	% Number of classes and prototypes

            % Initialize parameters

            self.Yh = -1*ones(Nc,N);
            self.winners = zeros(1,N);
            self.distances = zeros(Nk,N);

            if (self.nearest_neighbors == 1)
                self.nearest_indexes = zeros(1,N);
            else
                if (Nk <= self.nearest_neighbors)
                    self.nearest_indexes = zeros(Nk,N);
                else
                    self.nearest_indexes = zeros(self.nearest_neighbors+1,N);
                end
            end
            
            for n = 1:N
                
                % Display classification iteration (for debug)
                if(mod(n,1000) == 0)
                    disp(n);
                end
                
                % Predict (1 instance)
                self = self.partial_predict(X(:,n));
                
                % Hold values
                self.Yh(:,n) = self.yh;
                self.winners(n) = self.winner;
                self.distances(:,n) = self.distance;
                self.nearest_indexes(:,n) = self.nearest_index;
                
            end

        end

    end % end methods

    methods (Static)
        
        function winner = findWinnerPrototype(Cx,sample,self)

            [~,Nk] = size(Cx);
            Vdist = zeros(1,Nk);
            
            for i = 1:Nk
                Vdist(i) = vectorsDistance(Cx(:,i),sample,self);
            end

            if(self.distance == 0) % dot product
                [~,winner] = max(Vdist);
            else % other distance measures
                [~,winner] = min(Vdist);
            end

        end

    end

end % end class