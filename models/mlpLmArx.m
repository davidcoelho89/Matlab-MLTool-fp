classdef mlpLmArx < identifierArx
    %
    % --- MLP (with Levemberg Marquardt) for System Identification ---
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
        
        non_linearity = 'sigmoid';
        number_of_hidden_neurons = 5;
        minMSE = 1;
        minGRAD = 0.1;
        number_of_epochs = 1000;
        Muscale = 10;
        Mu_min = 1e-10;
        Mu_max = 1e+10;
        Mu_init = 0.01;
        
    end
    
    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        name = 'mlpLm';

        Win = [];
        Wout = [];
        bin = [];
        bout = [];
        out_layer1 = [];
        out_layer2 = [];
        Mu = [];
        J = [];
        error = [];
        
    end
    
    methods
        
        % Constructor
        function self = mlpLmArx()
            % Set the hyperparameters after initializing!
        end

        % Initialize Parameteres
        function self = initialize_parameters(self,x,y)
            
            % Needed for all arx models
            self.last_predictions_memory = x(1:sum(self.output_lags));
            
            p = length(x);
            No = length(y);

            self.Win = rand(self.number_of_hidden_neurons,p);
            self.bin = rand(self.number_of_hidden_neurons,1);
            self.Wout = rand(No,self.number_of_hidden_neurons);
            self.bout = rand(No,1);

            self.Mu = self.Mu_init;

            % Always consider bias
            self.add_bias = 0;
            
        end
        
        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            self = self.initialize_parameters(X(:,1),Y(:,1));

            % Init error and cost function
            self = self.calculate_output(X);
            self.Yh = self.yh;
            disp('size of Y');
            disp(size(Y));
            disp('size of Yh');
            disp(size(Yh));
            self.error = Y - self.Yh;
            costx = 0.5*sum(self.error.^2);

            epoch = 0;
            loop1 = 1;

            while loop1 && epoch < self.number_of_epochs

                epoch = epoch+1;                
                param = [self.Win(:); self.bin(:); self.Wout(:); self.bout(:)];
                self = self.mlpJacobian(self,X);
                
                loop2 = 1;
                while (loop2)

                    pk = -pinv(self.J'*self.J + self.Mu*eye(size(self.J,2),size(self.J,2)))*self.J'*self.error';
                    zk = param + pk;
                    self = self.devectorization(self,zk);
                    
                    self = self.calculate_output(X);
                    self.Yh = self.yh;
                    self.error = Y - self.Yh;
                    costz = 0.5*sum(self.error.^2);

                    if(costz < costx)
                        self.Mu = self.Mu/self.Muscale;
                        loop2 = 0;
                    else
                        self.Mu = self.Mu*self.Muscale;
                    end

                    if (self.Mu < self.Mu_min) || (self.Mu > self.Mu_max)
                        loop2 = 0;
                        disp('Hessian Matrix is singular')
                    end

                end % end while (loop2)

                costx = 0.5*sum(self.error.^2);
                clc;
                disp(costx);
                
                if (costx < self.minMSE) && (norm(2*self.J'*self.error') < self.minGRAD)
                    loop1 = 0;
                    disp('Minimum train error (MSE) and Gradient vector norm are satisfied');
                end

            end % end while (loop1)

            if epoch >= self.number_of_epochs
                disp('Max. iteration condition is satisfied');
            end

            self = self.calculate_output(X);
            self.Yh = self.yh;
        end
        
        % Need to be implemented for any ArxModel
        function number_of_outputs = get_number_of_outputs(self)
            [number_of_outputs,~] = size(self.Wout);
        end

        % Need to be implemented for any ArxModel
        function self = calculate_output(self,x)

            self.out_layer1 = self.Win*x + self.bin;
            Zi = (2./(1+exp(-2*self.out_layer1)))-1; % [-1,1] (tg hiperb)
            self.out_layer2 = self.Wout*Zi + self.bout;
            self.yh = purelin(self.out_layer2);
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
        
        function self = mlpJacobian(self,X)
            
            [~,p] = size(self.Win);
            [No,~] = size(self.Wout);
            [~,N] = size(X);

            self.J = zeros(N, self.Nhidden*(p+1+No) + No);

            for i = 1:N

                Zi = ((2./(1+exp(-2*self.out_layer1(:,i))))-1);
                
                dWout = -(1-tansig(self.out_layer2(:,i)).^2)*Zi; % [-1,1] (tg hiperb)
                % dWout = -1./(1+exp(-netin(:,i)));  % Saida entre [0,1] (sigmoide logistica)
                % dWout = -(1-exp(-netin(:,i)))./(1+exp(-netin(:,i))); % Saida entre [-1,1] (tg hp)
                % dWout = -((2./(1+exp(-2*netin(:,i))))-1); % Saida entre [-1,1] (tg hp)
                % dWout = -tansig(netin(:,i)

                dbout = -1;
                % dbout = sum(-(1-tansig(self.out_layer2(:,i)).^2));

                dWin = -((self.Wout'.*(1-tansig(self.out_layer1(:,i)).^2)))*X(:,i);
                dbin = sum(-self.Wout'.*(1-tansig(self.out_layer1(:,i)).^2),2);

                self.J(i,:) = [dWin(:); dbin(:); dWout(:); dbout(:)];

            end

        end % end mlpJacobian

        function self = devectorization(self,vector)

            [~,p] = size(self.Win);
            [No,~] = size(self.Wout);

            self.Win = reshape(vector(1:p*self.Nhidden), [self.Nhidden,p]);
            
            vector = vector(p*self.Nhidden+1:end);

            self.bin = reshape(vector(1:self.Nhidden), [self.Nhidden,1]);
            
            vector = vector(self.Nhidden+1:end);
    
            self.Wout = reshape(vector(1:No*self.Nhidden), [No,self.Nhidden]);
            
            vector = vector((No*self.Nhidden)+1:end);
    
            self.bout = reshape(vector(1:end),[No, 1]);

        end % end devectorization

    end % end static methods
    
end % end class













