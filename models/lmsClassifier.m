classdef lmsClassifier
    % HELP about lmsClassifier
    
    % Hyperparameters
    properties
        number_of_epochs = 200;
        learning_step = 0.05;
        video_enabled = 0;
        add_bias = 1;
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        W = [];     % Regression Matrix              [Nc x p] or [Nc x p+1]
        MQE = [];   % mean quantization error of training          [1 x Ne]
        video = []; % frame structure (can be played with 'video function')
    end
    
    methods
        
        % Constructor
        function self = lmsClassifier()
            % Set the hyperparameters after initializing!
        end
        
        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            
            if(isempty(self.W))
                if(self.add_bias == 1)
                    self.W = 0.01*rand(length(y),length(x)+1);
                else
                    self.W = 0.01*rand(length(y),length(x));
                end
            end
            
            if(self.add_bias)
                x = [1 ; x];
            end
            
            yh = self.W * x;
            e = y - yh;
            self.W = self.W + self.learning_step * e * x';
            
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
            
            if(self.add_bias)
                self.W = 0.01*rand(Nc,p+1);
            else
                self.W = 0.01*rand(Nc,p);
            end
            
            for ep = 1:self.number_of_epochs
                
                if(self.video_enabled)
                    self.video(ep) = hyperplane_frame(self.W,DATA);
                end
                
                % Shuffle Data
                I = randperm(N);
                X = X(:,I);
                Y = Y(:,I);
                
                for t = 1:N
                    self = self.partial_fit(X(:,t),Y(:,t));
                end
                
                Yh = self.predict(X);
                self.MQE(ep) = sum(sum((Y-Yh).^2))/N;
                
            end
        end
        
        % Prediction Function
        function Yh = predict(self,X)
            Yh = linearPrediction(self,X);
        end
        
    end
    
end