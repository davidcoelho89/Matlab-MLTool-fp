classdef elmArx < identifierArx
    %
    % --- ELM for System Identification ---
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
        number_of_hidden_neurons = 25;
        non_linearity = 'sigmoid';
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        W = [];
        structure = [];
        number_of_layers = [];
    end
    
    methods
        
        % Constructor
        function self = elmArx()
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
                self.W{1} = 0.01*(2*rand(self.number_of_hidden_neurons,p+1)-1);
            else
                self.W{1} = 0.01*(2*rand(self.number_of_hidden_neurons,p)-1);
            end
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [~,number_of_samples] = size(X);
            
            self = self.initialize_parameters(X(:,1),Y(:,1));
            
            if(self.add_bias)
                X = [ones(1,number_of_samples) ; X];
            end
            
            Xk = zeros(self.number_of_hidden_neurons+1,number_of_samples);
            for t = 1:number_of_samples
                xi = X(:,t);
                Ui = self.W{1} * xi;
                Yi = self.activation_function(Ui,self.non_linearity);
                Xk(:,t) = [1; Yi];
            end
            
            self.W{2} = Y*pinv(Xk);
            
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
























