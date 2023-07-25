classdef mlpArx < identifierArx
    %
    % --- MLP for System Identification ---
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
    %   - mlpArx()                  % Constructor 
    %   - predict(self,X)           % Prediction function (N instances)
    %   - partial_predict(self,x)   % Prediction function (1 instance)
    %
    %   - update_output_memory_from_prediction(self)
    %   - update_regression_vector_from_memory(self,x)
    %
    %   - self = initialize_parameters(self,x,y)
    %   - self = partial_fit(self,x,y)
    %   - self = fit(self,X,Y)
    %
    %   - self = calculate_output(self,x)
    %   - number_of_outputs = get_number_of_outputs(self)
    % 
    %   - Yi = activation_function(Ui,non_linearity)
    %   - Di = function_derivate(Yi,non_linearity)
    %
    % ----------------------------------------------------------------
    
    % Hyperparameters
    properties
        
        number_of_epochs = 200;
        number_of_hidden_neurons = 8;
        learning_rate = 0.05;
        moment_factor = 0.75;
        non_linearity = 'hyperbolic_tangent';
        video_enabled = 0;
        
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
        error = [];
        
    end
    
    methods
        
        % Constructor
        function self = mlpArx()
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
            self.W_old = cell(self.number_of_layers,1);

            for i = 1:self.number_of_layers
                self.W{i} = 0.01*rand(self.structure(i+1),self.structure(i)+1);
                self.W_old{i} = self.W{i};
            end
            
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            % Need this function for first instance
            if(isempty(self.W))
                self = self.initialize_parameters(x,y);
            end
            
            % Need this function for free simulation
            self = self.hold_last_output_from_fit(y);
                        
            % information of each layer
            local_input = cell(self.number_of_layers,1);
            local_output = cell(self.number_of_layers,1);
            local_gradient = cell(self.number_of_layers,1);
            
            % Forward Step (Calculate Layers' Outputs)
            for i = 1:self.number_of_layers
                
                if (i == 1) % input layer
                    if (self.add_bias)
                        local_input{i} = [+1; x];
                    else
                        local_input{i} = x;
                    end
                else % other layers (add bias to last output)
                    local_input{i} = [+1; local_output{i-1}];
                end
                
                % Neurons' Activation 
                Ui = self.W{i} * local_input{i}; 
                if (i == self.number_of_layers) % output_layer
                    local_output{i} = self.activation_function(Ui,'linear');
                else
                    local_output{i} = self.activation_function(Ui,self.non_linearity);
                end
            end
            
            % Error Calculation
            self.error = y - local_output{self.number_of_layers};
            
            % Backward Step (Calculate Layers' Local Gradients)
            for i = self.number_of_layers:-1:1
                
                if (i == self.number_of_layers) % output layer
                    Di = self.function_derivate(local_output{i},'linear');
                    local_gradient{i} = Di.*self.error;
                else
                    Di = self.function_derivate(local_output{i},self.non_linearity);
                    retropropagated_error = self.W{i+1}(:,2:end)'*local_gradient{i+1};
                    local_gradient{i} = Di.*retropropagated_error;
                end
            end
            
            % Update Model's Weights
            for i = self.number_of_layers:-1:1
                W_aux = self.W{i};
                self.W{i} = self.W{i} + ...
                            self.learning_rate*local_gradient{i}*local_input{i}' + ...
                            self.moment_factor*(self.W{i} - self.W_old{i});
                self.W_old{i} = W_aux;
            end
            
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [~,number_of_samples] = size(X);
            
            self = self.initialize_parameters(X(:,1),Y(:,1));
            self.MQE = zeros(1,self.number_of_epochs);
            self.video = struct('cdata',...
                                cell(1,self.number_of_epochs),...
                                'colormap',...
                                cell(1,self.number_of_epochs));
            
            for epoch = 1:self.number_of_epochs
                
                if(self.video_enabled)
                    self.video(epoch) = get_frame_hyperplane(self,X,Y);
                end
                
                % Shuffle Data
                I = randperm(number_of_samples);
                X = X(:,I);
                Y = Y(:,I);

                SQE = 0;
                
                for t = 1:number_of_samples
                      self = self.partial_fit(X(:,t),Y(:,t));
                      SQE = SQE + sum((self.error).^2);
                end
                
                self.MQE = SQE/number_of_samples;
                
            end
            
        end
        
        % Need to be implemented for any ArxModel
        function number_of_outputs = get_number_of_outputs(self)
            [number_of_outputs,~] = size(self.W{self.number_of_layers});
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
        
    end % end methods
    
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

        % Different types of non-linear functions derivates
        function Di = function_derivate(Yi,non_linearity)
            % There is a minimum of 0.05 out so as not to paralyze the learning
            if(strcmp(non_linearity,'linear'))
                Di = ones(size(Yi));
            elseif(strcmp(non_linearity,'sigmoid'))
                Di = Yi.*(1 - Yi) + 0.05;
            elseif(strcmp(non_linearity,'hyperbolic_tangent'))
                Di = 0.5*(1-Yi.^2) + 0.05;
            else
                Di = ones(size(Yi));
                disp('Invalid function option. Linear function chosen.');
            end
        end
        
    end % end static methods
    
end % end class