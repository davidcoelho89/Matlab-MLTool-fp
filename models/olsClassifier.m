classdef olsClassifier
    % HELP about olsClassifier
    % approximation:
	%	pinv -> W = Y*pinv(X);
    %	svd -> W = Y/X;
    %	theoretical -> W = Y*X'/(X*X' + regularization * eye(p,p));

    % Hyperparameters
    properties
        approximation = 'pinv';
        regularization = 0.0001;
        add_bias = 1;
        Yh = [];    % Hold all predictions
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        name = 'ols';
        W = [];     % Regression Matrix     [Nc x p] or [Nc x p+1]
    end
    
    methods
        
        % Constructor
        function self = olsClassifier()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (1 instance)
%         function self = partial_fit(self,x,y)
%             % Verify if an RLS can be used
%         end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [p,N] = size(X);
            if(self.add_bias)
                p = p+1;
                X = [ones(1,N) ; X];
            end
            
            if(strcmp(self.approximation,'pinv'))
                self.W = Y*pinv(X);
            elseif(strcmp(self.approximation,'svd'))
                self.W = Y/X;
            elseif(strcmp(self.approximation,'theoretical'))
                self.W = Y*X'/(X*X' + self.regularization * eye(p,p));
            else
                self.W = Y*pinv(X);
            end            
            
        end
        
        % Prediction Function
        function self = predict(self,X)
            self.Yh = linearPrediction(self,X);
        end
        
    end
    
end