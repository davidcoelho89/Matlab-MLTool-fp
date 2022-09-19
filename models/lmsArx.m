classdef lmsArx
    % HELP about lmsArx
    
    % Hyperparameters
    properties
        number_of_epochs = 05;
        learning_step = 0.05;
        add_bias = 1;
        video_enabled = 0;
        training_samples_count = 0;
        prediction_type = 1;
        output_lags = [];
        Yh = [];    % Hold all predictions
        yh = [];    % Hold last prediction
        output_memory = []; % hold value of last predictions
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        name = 'lms';
        W = [];     % Regression Matrix              [Nc x p] or [Nc x p+1]
        W_acc = []; % Accumulate progression of weights
        MQE = [];   % mean quantization error of training          [1 x Ne]
        video = []; % frame structure (can be played with 'video function')
    end
    
    methods
        
        % Constructor
        function self = lmsArx()
            % Set the hyperparameters after initializing!
        end
        
        % Initialize Parameteres
        function self = initialize_parameters(self,x,y)
            if(self.add_bias == 1)
                self.W = 0.01*rand(length(y),length(x)+1);
            else
                self.W = 0.01*rand(length(y),length(x));
            end
            self.output_memory = x(1:sum(self.output_lags));
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            if(isempty(self.W))
                self = initialize_parameters(x,y);
            end
            
            self.output_memory = update_output_memory(y,...
                                                      self.output_memory,...
                                                      self.output_lags);
            if(self.add_bias)
                x = [1 ; x];
            end
            
            % Update Model's Weights
            self.yh = self.W * x;
            e = y - self.yh;
            self.W = self.W + self.learning_step * e * x' / (x'*x);
            
            % Save weight update
            if(~isempty(self.W))
                self.training_samples_count = self.training_samples_count + 1;
                self.W_acc{1,self.training_samples_count} = self.W;
            end
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [~,number_of_samples] = size(X);
            
            self = self.initialize_parameters(X(:,1),Y(:,1));
            self.MQE = zeros(1,self.number_of_epochs);
            self.video = struct('cdata',...
                                cell(1,self.number_of_epochs), ...
                                'colormap', ...
                                cell(1,self.number_of_epochs) ...
                                );
            
            % Initialize W accumulator
            self.W_acc = cell(1,self.number_of_epochs*number_of_samples+1);
            self.W_acc{1,1} = self.W;
            
            for epoch = 1:self.number_of_epochs
                
                if(self.video_enabled)
                    self.video(epoch) = hyperplane_frame(self.W,DATA);
                end
                
                for n = 1:number_of_samples
                    self = self.partial_fit(X(:,n),Y(:,n));
                end
                
                Y_h = linearPrediction(self,X);
                self.MQE(epoch) = sum(sum((Y - Y_h).^2))/number_of_samples;
                
            end
        end
        
        % Prediction Function (1 instance)
        function self = partial_predict(self,x)
            x = update_regression_vector(x,...
                                         self.output_memory, ...
                                         self.prediction_type, ...
                                         self.add_bias);
            self.yh = self.W * x;
            
            self.output_memory = update_output_memory(self.yh,...
                                                      self.output_memory,...
                                                      self.output_lags);
        end
        
        % Prediction Function (N instances)
        function self = predict(self,X)
            
            [~,number_of_samples] = size(X);
            [number_of_outputs,~] = size(self.W);
            
            self.Yh = zeros(number_of_outputs,number_of_samples);
            
            if(self.add_bias)
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
                
                self = self.partial_predict(X(:,n));
                self.Yh(:,n) = self.yh;
                
%                 xn = X(:,n);
%                 
%                 xn = update_regression_vector(xn,...
%                                               self.output_memory, ...
%                                               self.prediction_type, ...
%                                               self.add_bias);
% 
%                 self.Yh(:,n) = self.W * xn;
%                 
%                 self.output_memory = update_output_memory(self.Yh(:,n),...
%                                                           self.output_memory,...
%                                                           self.output_lags);
            end
            
        end
        
    end
    
end