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
        output_lags = 2;
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
        yh = [];
        error = [];
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

            for i = 1:self.number_of_layers
                self.W{i} = 0.01*rand(self.structure(i+1),self.structure(i)+1);
                self.W_old{i} = self.W{i};
            end
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            if(isempty(self.W))
                self = self.initialize_parameters(self,x,y);
            end
            
            x_layer = cell(self.number_of_layers,1); % input of each layer
            y_layer = cell(self.number_of_layers,1); % output of each layer
            delta = cell(self.number_of_layers,1); % local gradient of each layer
            
            % Forward Step (Calculate Layers' Outputs)
            for i = 1:self.number_of_layers
                
                if (i == 1) % input layer
                    if (self.add_bias)
                        x_layer{i} = [+1; x];
                    else
                        x_layer{i} = x;
                    end
                else % other layers (add bias to last output)
                    x_layer{i} = [+1; y_layer{i-1}];
                end
                
                % Activation of hidden neurons
                Ui = self.W{i} * x_layer{i}; 
                
                if (i == self.number_of_layers)
                    y_layer{i} = mlp_f_ativ(Ui,'linear');
                else
                    y_layer{i} = mlp_f_ativ(Ui,self.non_linearity);
                end
            end
            
            % Error Calculation
            self.error = y - y_layer{self.number_of_layers};
            
            % Backward Step (Calculate Layers' Local Gradients)
            for i = self.number_of_layers:-1:1
                f_der = mlp_f_gradlocal(y_layer{i},self.non_linearity);
                if (i == self.number_of_layers) % output layer
                    % delta{i} = self.error.*f_der;
                    delta{i} = self.error; % linear function.
                else
                    delta{i} = f_der.*(self.W{i+1}(:,2:end)'*delta{i+1});
                end
            end
            
            % Weight Adjustment
            for i = self.number_of_layers:-1:1
                W_aux = self.W{i};
                self.W{i} = self.W{i} + ...
                            self.learning_rate*delta{i}*x_layer{i}' + ...
                            mom*(self.W{i} - self.W_old{i});
                self.W_old{i} = W_aux;
            end
            
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [~,number_of_samples] = size(X);
            
            self = self.initialize_parameters(self,X(:,1),Y(:,1));
            self.MQE = zeros(1,self.number_of_epochs);
            self.video = struct('cdata',cell(1,Nep),'colormap', cell(1,Nep));
            
            for epoch = 1:self.number_of_epochs
                
                if(self.video_enabled)
                    self.video(epoch) = get_frame_hyperplane(self,X,Y);
                end
                
                % Shuffle Data
                I = randperm(number_of_samples);
                X = X(:,I);
                Y = Y(:,I);

                SQE = 0; % Init sum of quadratic errors
                
                for t = 1:number_of_samples
                      self = self.partial_fit(self,X(:,t),Y(:,t));
                      SQE = SQE + sum((self.error).^2);
                end
                
                self.MQE = SQE/number_of_samples;
                
            end
            
        end
        
        % Prediction Function
        function self = predict(self,X)
            
            [number_of_outputs,~] = size(self.W{self.number_of_layers});
            [~,number_of_samples] = size(X);
            
            self.yh = zeros(number_of_outputs,number_of_samples);
            
            if(self.add_bias == 1)
                X = [ones(1,number_of_samples) ; X];
            end
            
            % Initialize memory of last predictions (for free simulation)
            output_memory_length = sum(self.output_lags);
            if(self.add_bias)
                self.output_memory = X(2:output_memory_length+1,1);
            else
                self.output_memory = X(1:output_memory_length,1);
            end
            
            for n = 1:number_of_samples
                
                xn = X(:,n);
                
                xn = update_regression_vector(xn,...
                                              self.output_memory,...
                                              self.prediction_type, ...
                                              self.add_bias);

                for i = 1:self.number_of_layers
                    Ui = self.W{i} * xn;
                    if i == self.number_of_layers
                        Yi = mlp_f_ativ(Ui,0);
                    else
                        Yi = mlp_f_ativ(Ui,self.non_linearity);
                    end
                    xn = [+1; Yi];
                end
                
                self.yh(:,n) = Yi;
                
                self.output_memory = update_output_memory(Yi,...
                                                          self.output_memory,...
                                                          self.output_lags);
                
            end
            
        end
        
    end
    
end





