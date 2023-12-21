classdef linearClassifier
    %
    % --- Common features for all Linear Classifiers ---
    %
    % Properties (Hyperparameters)
    %
    %   - regularization = used to mitigate numerical computation errors
    %     (constant)
    %   - add_bias = add bias (or not) to the model
    %     (0 or 1)
    %
    % Properties (Parameters)
    %
    %   - W = regression matrix [Nc x p] or [Nc x p+1]
    %   - Yh = Hold all predictions [Nc x N]
    %
    % Methods
    %
    %   - linearClassifier()        % Constructor
    %   - predict(self,X)           % Prediction Function
    %
    % ----------------------------------------------------------------
    
    % Hyperparameters
    properties
        regularization = 0.0001;
        add_bias = 1;
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        W = [];    
        Yh = [];
    end

    methods

        % Constructor
        function self = linearClassifier()
            % Set the hyperparameters after initializing
        end

        % Training Function (1 instance)
        % function self = partial_fit(self,x,y)
        %     % Each classifier has it own function
        % end

        % Training Function (N instances)
        % function self = fit(self,X,Y)
        %   "Each classifier has it own function"
        % end

        % Prediction Function
        function self = predict(self,X)
            [~,N] = size(X);
            if(self.add_bias)
                X = [ones(1,N) ; X];
            end
            self.Yh = self.W * X;
        end

    end % end methods

end % end class