classdef lmsArx < identifierArx
    %
    % --- LMS for System Identification ---
    %
    % Properties (Hyperparameters)
    %
    %   - add_bias = 0 or 1. Add bias or not.
    %   - prediction_type "=0": free simulate. ">0": n-steps ahead
    %   - output_lags = indicates number of lags for each output
    %   - number_of_epochs = integer >= 1;
    %   - learning_step = real number between 0 and 1
    %   - video_enabled = 0 or 1;
    %   - training_samples_count = integer. for debug.;
    %
    % Properties (Parameters)
    %
    %   - Yh = matrix that holds all predictions  [Noutputs x Nsamples]
    %   - yh = vector that holds last prediction  [Noutputs x 1]
    %   - last_predictions_memory = vector holding past values of predictions
    %
    %   - W = Regression Matrix [Nc x p] or [Nc x p+1]
    %   - W_acc = Accumulate progression of weights
    %   - MQE = mean quantization error of training [1 x Ne]
    %   - video = frame structure (can be played with 'video function')
    %
    % Methods
    %
    %   - self = lmsArx()
    %   - self = predict(self,X) % Prediction function (N samples)
    %   - self = partial_predict(self,x) % Prediction function (1 sample)
    %
    %   - self = update_output_memory_from_prediction(self)
    %   - xn_out = update_regression_vector_from_memory(self,x)
    %
    %   - self = initialize_parameters(self,x,y)
    %   - self = partial_fit(self,x,y) % Training Function (1 sample)
    %   - self = fit(self,X,Y) % Training Function (N samples)
    %
    %   - self = calculate_output(self,x)
    %   - number_of_outputs = get_number_of_outputs(self)
    % 
    % ----------------------------------------------------------------
    
    % Hyperparameters
    properties
        
        number_of_epochs = 05;
        learning_step = 0.05;
        video_enabled = 0;
        training_samples_count = 0;
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        name = 'lms';
        W = [];
        W_acc = [];
        MQE = [];
        video = [];
        
    end
    
    methods
        
        % Constructor
        function self = lmsArx()
            % Set the hyperparameters after initializing!
        end
        
        % Initialize Parameteres
        function self = initialize_parameters(self,x,y)
            
            % Needed for all arx models
            self.last_predictions_memory = x(1:sum(self.output_lags));

            if(self.add_bias == 1)
                self.W = 0.01*rand(length(y),length(x)+1);
            else
                self.W = 0.01*rand(length(y),length(x));
            end
            
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            % Need this function for first instance
            if(isempty(self.W))
                self = initialize_parameters(x,y);
            end
            
            % Need this function for free simulation
            self = self.hold_last_output_from_fit(y);
            
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
                
                self = self.predict(X);
                self.MQE(epoch) = sum(sum((Y - self.Yh).^2))/number_of_samples;
                
            end
        end
        
        % Need to be implemented for any ArxModel
        function self = calculate_output(self,x)
            self.yh = self.W * x;
        end

        % Need to be implemented for any ArxModel
        function number_of_outputs = get_number_of_outputs(self)
            [number_of_outputs,~] = size(self.W);
        end
        
    end % end methods
    
end % end class