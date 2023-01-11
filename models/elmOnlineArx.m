classdef elmOnlineArx < identifierArx
    %
    % --- ELM for Online System Identification ---
    %
    % Properties (Hyperparameters)
    %
    %   - add_bias = 0 or 1. Add bias or not.
    %   - prediction_type "=0": free simulate. ">0": n-steps ahead
    %   - output_lags = indicates number of lags for each output
    %
    % Properties (Parameters)
    %
    %   - Yh = matrix that holds all predictions  [Noutputs x Nsamples]
    %   - yh = vector that holds last prediction  [Noutputs x 1]
    %   - last_predictions_memory = vector holding past values of predictions
    %
    % Methods
    %
    %   - self = elmArx()
    %   - self = predict(self,X) % Prediction function (N samples)
    
    % Hyperparameters
    properties
        forgiving_factor = 1;
        number_of_hidden_neurons = 25;
        non_linearity = 'sigmoid';
        number_of_epochs = 1;
        P_init = 10000;
        W_init = 0.01;
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        W = [];
        P = [];
        structure = [];
        number_of_layers = [];
    end
    
    methods
        
        % Constructor
        function self = elmOnlineArx()
            % Set the hyperparameters after initializing!
        end
        
        % Initialize Parameteres
        function self = initialize_parameters(self,x,y)
            
            % Needed for all arx models
            self.last_predictions_memory = x(1:sum(self.output_lags));
            
            p = length(x);
            No = length(y);
            
            self.structure = [p,self.number_of_hidden_neurons,No];
            
            self.number_of_layers = length(self.structure) - 1;
            
            self.W = cell(self.number_of_layers,1);
            
            if(self.add_bias)
                self.W{1} = self.W_init*(2*rand(self.number_of_hidden_neurons,p+1)-1);
            else
                self.W{1} = self.W_init*(2*rand(self.number_of_hidden_neurons,p)-1);
            end
            self.W{2} = zeros(No,self.number_of_hidden_neurons + 1);
            self.P = self.P_init * eye(self.number_of_hidden_neurons + 1);

        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            % Need this function for first instance
            if(isempty(self.W))
                self = self.initialize_parameters(x,y);
            end
            
            % Need this function for free simulation
            self = self.hold_last_output_from_fit(y);
            
            % Add bias to input
            if(self.add_bias)
                x = [1 ; x];
            end
            
            % Get output from hidden layer (input for output layer)
            Ui = self.W{1} * x;
            Yi = self.activation_function(Ui,self.non_linearity);
            xk = [1; Yi];
            
            % Update W{2}
            K = self.P*xk/(self.forgiving_factor + xk'*self.P*xk);
            error = (y' - xk'*self.W{2}');
            self.W{2} = self.W{2} + error'*K';
            self.P = (1/self.forgiving_factor)*(self.P - K*xk'*self.P);
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [~,number_of_samples] = size(X);
            
            self = self.initialize_parameters(X(:,1),Y(:,1));
            
            for epoch = self.number_of_epochs
                for n = 1:number_of_samples
                    self = self.partial_fit(X(:,n),Y(:,n));
                end
            end
            
        end
        
        % Need to be implemented for any ArxModel
        function self = calculate_output(self,x)

            for i = 1:self.number_of_layers
                Ui = self.W{i} * x;
                if i == self.number_of_layers % output layer
                    Yi = self.activation_function(Ui,'linear');
                else
                    Yi = self.activation_function(Ui,self.non_linearity);
                end
                x = [+1; Yi];
            end
            
            self.yh = Yi;
            
        end
        
        % Need to be implemented for any ArxModel
        function number_of_outputs = get_number_of_outputs(self)
            [number_of_outputs,~] = size(self.W{self.number_of_layers});
        end
        
    end
    
    methods (Static)
        
        % Different types of non-linear functions
        function Yi = activation_function(Ui,non_linearity)
            if(strcmp(non_linearity,'linear'))
                Yi = Ui;
            elseif(strcmp(non_linearity,'sigmoid'))
                Yi = 1./(1+exp(-Ui));
            elseif(strcmp(non_linearity,'hyperbolic_tangent'))
                Yi = (1-exp(-Ui))./(1+exp(-Ui));
            else
                Yi = Ui;
                disp('Invalid function option. Linear function chosen.');
            end
        end
        
    end
    
end