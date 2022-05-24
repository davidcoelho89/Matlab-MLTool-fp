classdef mlpArx
    % HELP about lmsArx
    
    % Hyperparameters
    properties
        number_of_epochs = 200;
        number_of_hidden_neurons = 8;
        learning_rate = 0.05;
        moment_factor = 0.75;
        non_linearity = 'sigmoid';
        add_bias = 1;
        video_enabled = 0;
        prediction_type = 1;
        output_lags = [];
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        name = 'mlp';
        W = [];
        MQE = [];
        video = [];
        output_memory = [];
    end
    
    methods
        
        % Constructor
        function self = mlpArx()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            self = self + x + y;
            
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            self = self + X + Y;
            
        end
        
        % Prediction Function
        function Yh = predict(self,X)
            
            Yh = self + X;
            
        end
        
    end
    
end