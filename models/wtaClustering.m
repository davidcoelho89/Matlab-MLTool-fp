classdef wtaClustering
    %
    % --- Winner Takes-All Clustering Function ---
    %
    % Properties (Hyperparameters)
    %  
    %   - distance_measure = which measure used to compare two vectors
    %      (used in a lot of functions for prototype-based classifiers)
    %   - kernel_type = which kernel used (kernel based classifiers)
    %       = 'none' (it is not a kernel based model)
    %   - initialization_type = how the prototypes will be initialized
    %       = 'zeros' (mean of normalized data)
    %       = 'random_samples' (forgy method. randomly choose k 
    %                           observations from data set)
    %       = 'calculate_centers' (Randomly assign a cluster to each 
    %                          observation, than update clusters' centers)
    %       = 'random_attributes' (prototype's attvalues randomly choosed 
    %                    between min and max values of data's attributes)
    %
    % Properties (Parameters)
    %
    %   - Cx = Clusters' centroids (prototypes)
    %	- winners = The closest prototype for each sample [1 x N]
    %   - winner = Closest prototypes to a sample [1 x 1]
    %	- distances = Distance from prototypes to each sample [Nk x N]
    %   - distance = Distance from prototypes a sample [Nk x 1]
    %   - Yh = All predictions (predict function) [1 x N]
    %   - yh = Last prediction (partial_predict function) [1 x 1]
    %   - video_structure = played by 'video function' [1 x Nep]
    %
    % Methods
    %
    %   - wtaClustering()       % Constructor
    %   - partial_fit(self,x,y) % fit Function (1 instance)
    %   - fit(self,X,Y)         % fit Function (N instances)
    %   - predict(self,X)       % Prediction Function
    %
    % ----------------------------------------------------------------

    % Hyperparameters
    properties
        distance_measure = 2;
        kernel_type = 'none';
        number_of_epochs = 200;
        number_of_prototypes = 20;
        initialization_type = 'random_samples';
        learning_type = 2;
        learning_step_initial = 0.7;
        learning_step_final = 0.01;
        video_enabled = 0;
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Cx = [];
        distances = [];
        distance = [];
        Yh = [];
        yh = [];
        video_structure = [];
        sum_of_squared_errors = [];
    end

    methods

        % Constructor
        function self = wtaClustering()
            % Set the hyperparameters after initializing!
        end

        % Initialize Parameteres
        function self = initialize_prototypes(self,X)

            [p,N] = size(X);
            self.Cx = zeros(p,self.number_of_prototypes);

            if(strcmp(self.initialization_type,'zeros'))
                % Does nothing: already initialized with zeros 
            elseif(strcmp(self.initialization_type,'random_samples'))
                I = randperm(N);
                self.Cx = X(:,I(1:Nk));
            elseif(strcmp(self.initialization_type,'calculate_centers'))
                samples_in_cluster = zeros(1,self.number_of_prototypes);
                I = rand(1,N);
                index = ceil(self.number_of_prototypes*I);
                for i = 1:N
                    samples_in_cluster(index(i)) = samples_in_cluster(index(i)) + 1;
                    self.Cx(:,index(i)) = self.Cx(:,index(i)) + X(:,i);
                end
                for i = 1:self.number_of_prototypes
                    self.Cx(:,i) = self.Cx(:,i) / samples_in_cluster(i);
                end
            elseif(strcmp(self.initialization_type,'random_attributes'))
                % Calculate min and max value of parameters
                [pmin,~] = min(X,[],2);
                [pmax,~] = max(X,[],2);
                % generate vectors
                for i = 1:Nk
                    self.Cx(:,i) = pmin + (pmax - pmin).*rand(p,1);
                end
            else
                disp('Unknown initialization. Prototypes = 0.');
            end

        end

        % Training Function (N instances)
        function self = fit(self,X)

            [~,N] = size(X);

            tmax = N*self.number_of_epochs;
            t = 0;
            
            self = self.initialize_prototypes(X);
            self.distances = zeros(self.number_of_prototypes,N);
            self.sum_of_squared_errors = zeros(1,self.number_of_epochs);
            self.video_structure = struct('cdata',...
                                          cell(1,self.number_of_epochs),...
                                          'colormap',...
                                          cell(1,self.number_of_epochs));

            for epoch = 1:self.number_of_epochs

                if(self.video_enabled)
                    self.video_structure(epoch) = get_prototypes_frame(self.Cx,X);
                end

                % Shuffle Data
                I = randperm(N);
                X = X(:,I);

                % Get Winner Neuron, update Learning Step, update prototypes
                for i = 1:N
                    t = t+1;
                    n = calculateLearningStep(self,tmax,t);
                    winner = self.findWinnerPrototype(X(:,i));
                    self.Cx(:,winner) = self.Cx(:,winner) + ...
                                        n * (X(:,i) - self.Cx(:,winner));
                end

                self.sum_of_squared_errors(epoch) = self.calculate_sse(self,X);

            end

        end

        % Prediction Function (1 instance)
        function self = partial_predict(self,x)
            % Todo - All
            self = self + x;
        end

        % Prediction Function (N instances)
        function self = predict(self,X)
            % Todo - All
            self = self + X;
        end
        
    end % end methods

    methods (Static)
        
        function winner = findWinnerPrototype(self,sample)

            [~,Nk] = size(self.Cx);
            Vdist = zeros(1,Nk);
            
            for i = 1:Nk
                Vdist(i) = vectorsDistance(self.Cx(:,i),sample,self);
            end

            if(self.distance == 0) % dot product
                [~,winner] = max(Vdist);
            else % other distance measures
                [~,winner] = min(Vdist);
            end

        end

        function sse = calculate_sse(self,X)
            % ToDo - all
            sse = self + X;
        end

        function n = calculateLearningStep(self,t,tmax)
            n = self + t + tmax;
        end

    end

end % end class