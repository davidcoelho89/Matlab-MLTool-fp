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
        W_old = [];
        structure = [];
        number_of_layers = [];
        MQE = [];
        video = [];
        output_memory = [];
    end
    
    methods
        
        % Constructor
        function self = mlpArx()
            % Set the hyperparameters after initializing!
        end

        % Initialize Parameteres
        function self = initialize_parameters(self,x,y)
            p = length(x);
            No = length(y);

            self.structure = [p,self.number_of_hidden_neurons,No];
            self.number_of_layers = length(self.structure) - 1;

            self.W = cell(self.number_of_layers,1);
            self.W_old = cell(self.number_of_layers,1);

            
            
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