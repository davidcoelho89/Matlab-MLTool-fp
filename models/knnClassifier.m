classdef knnClassifier
    %
    % --- K-Nearest Neighbors Classifier ---
    %
    % Properties (Hyperparameters)
    %
    % Properties (Parameters)
    %
    % Methods
    %
    %   - knnClassifier()           % Constructor
    %   - fit(self,X,Y)             % Training function (N instances)
    %   - predict(self,X)           % Prediction Function

    % Hyperparameters
    properties

    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Cx = [];     % Clusters' centroids (prototypes)
        Cy = [];     % Clusters' labels
    end

    methods

        % Constructor
        function self = knnClassifier()
            % Set the hyperparameters after initializing!
        end

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


















