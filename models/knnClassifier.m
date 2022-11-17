classdef knnClassifier < prototypeBasedClassifier
    %
    % --- K-Nearest Neighbors Classifier ---
    %
    % Properties (Hyperparameters)
    %  
    %   - distance_measure = which measure used to compare two vectors
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
    %   - Yh = all predictions (predict function)
    %
    % Methods
    %
    %   - knnClassifier()           % Constructor
    %   - fit(self,X,Y)             % Training function (N instances)
    %   - predict(self,X)           % Prediction Function

    % Hyperparameters
    properties
        distance_measure = 2;
        nearest_neighbors = 1;
        knn_aproximation = 'majority_voting';
        kernel_type = 'none';
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Cx = [];     % Clusters' centroids (prototypes)
        Cy = [];     % Clusters' labels
        Yh = [];	 % all predictions (predict function)
    end

    methods

        % Constructor
        function self = knnClassifier()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (1 instance)
%         function self = partial_fit(self,x,y)
%             % ToDo - Add one sample to existing prototypes
%         end

        % Training Function (N instances)
        function self = fit(self,X,Y)
            self.Cx = X;
            self.Cy = Y;
        end

        % Prediction Function (N instances)
        function self = predict(self,X)
            self = prototypesClassify(self,X);
        end

    end

end