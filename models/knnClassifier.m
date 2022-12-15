classdef knnClassifier < prototypeBasedClassifier
    %
    % --- K-Nearest Neighbors Classifier ---
    %
    % Properties (Hyperparameters)
    %  
    %   - distance_measure = which measure used to compare two vectors
    %       (used in a lot of functions for prototype-based classifiers)
    %       = 1 (manhattan) ; = 2 (euclidean) ; > 2 (minkowski)
    %       = Inf (Chebyshev) ; -Inf (smallest value) ; ...
    %   - nearest_neighbors = number of nearest neighbors 
    %      (for classification or regression)
    %   - knn_aproximation = how the output will be generated
    %       = 'majority_voting'
    %       = 'weighted_knn'
    %   - kernel_type = which kernel used (kernel based classifiers)
    %       = 'none' (it is not a kernel based model)
    %       = 'linear', 'gaussian', 'polynomial', 'exponential',
    %         'cauchy', 'log', 'sigmoid', 'kmod'
    %
    % Properties (Parameters)
    %
    %   - Cx = Clusters' centroids (prototypes)
    %   - Cy = Clusters' labels
    %   - Yh = all predictions (predict function)
    %	- winners = The closest prototype for each sample
    %	- distances = Distance from prototypes to each sample
    %	- nearest_indexes = identify nearest prototypes for each sample
    %
    % Methods
    %
    %   - knnClassifier()           % Constructor
    %   - fit(self,X,Y)             % Training function (N instances)
    %   - partial_fit(self,x,y)     % Training function (1 instance)
    %   - predict(self,X)           % Prediction Function
    %
    % ----------------------------------------------------------------

    % Hyperparameters
    properties
        
    % Following properties already defined in "prototypeBasedClassifier":
    % distance_measure = 2;
    % nearest_neighbors = 1;
    % knn_aproximation = 'majority_voting';
    % kernel_type = 'none';

    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
    % Following properties already defined in "prototypeBasedClassifier":
    % Cx = [];              % Clusters' centroids (prototypes)
    % Cy = [];              % Clusters' labels
    % Yh = [];	            % all predictions (predict function)
    % winners = [];         % The closest prototype for each sample
    % distances = [];       % Distance from prototypes to each sample
    % nearest_indexes = []; % identify nearest prototypes for each sample

    end

    methods

        % Constructor
        function self = knnClassifier()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            self.Cx = [self.Cx, x];
            self.Cy = [self.Cy, y];
        end

        % Training Function (N instances)
        function self = fit(self,X,Y)
            self.Cx = X;
            self.Cy = Y;
        end

        % Prediction Function (N instances)
        % function self = predict(self,X)
        %  This functions is already defined in "PrototypeBasedClassifier"
        % end

    end % end methods

end % end class