classdef prototypeBasedClassifier

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
        Yh = [];	          % all predictions (predict function)
        winners = [];         % The closest prototype for each sample
        distances = [];       % Distance from prototypes to each sample
        nearest_indexes = []; % identify nearest prototypes for each sample
    end

    methods

        % Constructor
        function self = prototypeBasedClassifier()
            % Set the hyperparameters after initializing!
        end

        % Training Function (1 instance)
%         function self = partial_fit(self,x,y)
%             % ToDo - Add one sample to existing prototypes
%             % (verify if RLS can be used)
%         end

        % Training Function (N instances)
        function self = fit(self,X,Y)
            self.Cx = X;
            self.Cy = Y;
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

            % 1 Nearest Neighbor Case
            if (self.nearest_neighbors == 1)
                
                for n = 1:N
                
                    % Display classification iteration (for debug)
                    if(mod(n,1000) == 0)
                        disp(n);
                    end
                
                    % Get test sample
                    sample = X(:,n);
    
                    % Get closest prototype and min distance 
                    % from sample to each class
                    
                    d_min = -1*ones(Nc,1);  % Init class min distance
                    d_min_all = -1;         % Init global min distance
    
                    for k = 1:Nk % for k = 1:Nk(1)
                        
                        prototype = self.Cx(:,k);
                        [~,class] = max(self.Cy(:,k));
                        
                        d = vectorsDistance(prototype,sample,self);
                        self.distances(k,n) = d;
                        
                        % Get class-conditional closest prototype
                        if(d_min(class) == -1 || d < d_min(class))
                            d_min(class) = d;
                        end
            
                        % Get global closest prototype
                        if(d_min_all == -1 || d < d_min_all)
                            d_min_all = d;
                            self.winners(n) = k;
                            self.nearest_indexes(n) = k;
                        end
            
                    end
    
                    % Fill Output
    
                    for class = 1:Nc
    
                        % Invert signal for 2nd class in binary problems
                        if(class == 2 && Nc == 2)
            	            self.Yh(2,:) = -self.Yh(1,:);
                            break;
                        end
    
                        % Calculate Class Output for the Sample
            
                        % Get minimum distance from class
                        dp = d_min(class);
                        % There is no prototypes from this class
                        if (dp == -1)
                            self.Yh(class,n) = -1;
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
                                elseif (dm == -1 || d_min(j) < dm)
                                    dm = d_min(j);
                                end
                            end
                            % no prototypes from other classes
                            if (dm == -1)  
                                self.Yh(class,n) = 1;
                            else
                                self.Yh(class,n) = (dm - dp) / (dm + dp);
                            end
                        end
    
                    end % end for class = 1:Nc

                end % end for n = 1:N

            % K Nearest neighbors Case
            elseif(self.nearest_neighbors > 1)

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
                        prototype = self.Cx(:,k);
                        Vdist(k) = vectorsDistance(prototype,sample,self);
                    end

                    % hold distances
                    self.distances(:,n) = Vdist';

                    % Sort distances and get nearest neighbors
                    sort_result = bubble_sort(Vdist,1);
            
                    % Get closest prototype
                    self.winners(n) = sort_result.ind(1);

                    % Verify number of prototypes and neighbors
                    if(Nk <= K)
                        self.nearest_indexes(:,n) = sort_result.ind(1:Nk)';
                        number_of_nearest = Nk;
                    else
                        self.nearest_indexes(:,n) = sort_result.ind(1:K+1)';
                        number_of_nearest = K;
                    end

                    % Get labels of nearest neighbors
                    lbls_near = model.Cy(:,self.nearest_indexes(:,n)');

                    if(strcmp(self.knn_aproximation,'majority_voting'))

                        % Compute votes
                        votes = zeros(1,Nc);
                        for k = 1:number_of_nearest
                            [~,class] = max(lbls_near(:,k));
                            votes(class) = votes(class) + 1;
                        end
                        
                        % Update class
                        [~,class] = max(votes);
                        self.Yh(class,n) = 1;

                    else % Weighted KNN
            
                        % Avoid weights of 0
                        epsilon = 0.001;
            
                        % Auxiliary output and weight
                        y_aux = zeros(Nc,1);
                        w_sum = 0;
                        
                        % Get distances of nearest neighbors
                        Dnear = Vdist(self.nearest_indexes(:,n)');
                        
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
                        self.Yh(:,n) = y_aux / w_sum;

                    end % end if majority_voting

                end % end for n = 1:N

            end % end if(self.nearest_neighbors == 1)

        end % end predict function

    end % end methods

end