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
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
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
            
            % Get Number of samples, attributes and classes
            [Nc,~] = size(Y);
            [p,N] = size(X);
            
            if(self.add_bias)
                self.W = 0.01*rand(Nc,p+1);
            else
                self.W = 0.01*rand(Nc,p);
            end
            
            X = [ones(1,N) ; X];
            
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
        function Yh = predict(self,X)
            Yh = linearPrediction(self,X);
        end
        
    end
    
end