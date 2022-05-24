classdef lmsArx
    % HELP about lmsArx
    
    % Hyperparameters
    properties
        number_of_epochs = 200;
        learning_step = 0.05;
        add_bias = 1;
        video_enabled = 0;
        training_samples_count = 0;
        prediction_type = 1;
        output_lags = [];
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        name = 'lms';
        W = [];     % Regression Matrix              [Nc x p] or [Nc x p+1]
        W_acc = []; % Accumulate progression of weights
        MQE = [];   % mean quantization error of training          [1 x Ne]
        video = []; % frame structure (can be played with 'video function')
        output_memory = []; % hold value of last predictions
        Yh = [];    % Hold all predictions
    end
    
    methods
        
        % Constructor
        function self = lmsArx()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            % Initialize W (if needed)
            if(isempty(self.W))
                if(self.add_bias == 1)
                    self.W = 0.01*rand(length(y),length(x)+1);
                else
                    self.W = 0.01*rand(length(y),length(x));
                end
            end
            
            % Update output memory
            self.output_memory = x(1:sum(self.output_lags));
            
            % Add bias if needed
            if(self.add_bias)
                x = [1 ; x];
            end
            
            % Update Model's Weights
            yh = self.W * x;
            e = y - yh;
            self.W = self.W + self.learning_step * e * x' / (xn'*xn);
            
            % Save weight update
            if(~isempty(self.W))
                self.training_samples_count = self.training_samples_count + 1;
                self.W_acc{1,self.training_samples_count} = self.W;
            end
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            % Get Number of samples, attributes and classes
            [Nc,~] = size(Y);
            [p,N] = size(X);
            
            self.MQE = zeros(1,self.number_of_epochs);
            
            self.video = struct('cdata',...
                                cell(1,self.number_of_epochs), ...
                                'colormap', ...
                                cell(1,self.number_of_epochs) ...
                                );
            
            % Initialize W
            if(self.add_bias)
                self.W = 0.01*rand(Nc,p+1);
            else
                self.W = 0.01*rand(Nc,p);
            end
            
            % Initialize W accumulator
            self.W_acc = cell(1,self.number_of_epochs*N+1);
            self.W_acc{1.1} = self.W;
            
            for ep = 1:self.number_of_epochs
                
                if(self.video_enabled)
                    self.video(ep) = hyperplane_frame(self.W,DATA);
                end
                
                for n = 1:N
                    self = self.partial_fit(X(:,n),Y(:,n));
                end
                
                Y_h = linearPrediction(self,X);
                self.MQE(ep) = sum(sum((Y-Y_h).^2))/N;
                
            end
        end
        
        % Prediction Function
        function self = predict(self,X)
            
            [~,N] = size(X);
            [Nc,~] = size(self.W);
            
            self.Yh = zeros(Nc,N);
            
            for n = 1:N
                xn = X(:,n);
                
                xn = update_regression_vector(xn,...
                                              self.output_memory,...
                                              self.prediction_type, ...
                                              self.add_bias);
    
                self.Yh(:,n) = linearPrediction(self,xn);
                
                self.output_memory = update_output_memory(self.Yh(:,n),...
                                                          self.output_memory,...
                                                          self.output_lags);
            end
            
        end
        
    end
    
end