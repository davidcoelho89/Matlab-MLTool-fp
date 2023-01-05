classdef rlsArx < identifierArx
    %
    % --- RLS for System Identification ---
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
    %   - self = rlsArx()
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

        forgiving_factor = 1;
        video_enabled = 0;

    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)

        name = 'rls';
        W = [];
        W_acc = [];
        P = [];
        P_acc = [];
        MQE = [];
        video = [];
    
    end

    methods
    
        % Constructor
        function self = rlsArx()
            % Set the hyperparameters after initializing!
        end

        % Initialize Parameteres
        function self = initialize_parameters(self,x,y)
            
            % Needed for all arx models
            self.last_predictions_memory = x(1:sum(self.output_lags));

            % W(0) and P(0). W(0) Could be 0.01*rand(C,P)
            if(self.add_bias == 1)
                self.W = zeros(length(x)+1,length(y));
                self.P = 1e+4 * eye(length(x)+1);
            else
                self.W = zeros(length(x),length(y));
                self.P = 1e+4 * eye(length(x));
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

            if(self.add_bias)
                x = [1 ; x];
            end

            % Update weights
            K = self.P*x/(self.forgiving_factor + x'*self.P*x);
            error = (y' - x'*self.W);
            self.W = self.W + K*error;
            self.P = (1/self.forgiving_factor)*(self.P - K*x'*self.P);
            
        end

        % Training Function (N instances)
        function self = fit(self,X,Y)
            
            [~,number_of_samples] = size(X);

            self = self.initialize_parameters(X(:,1),Y(:,1));

            self.video = struct('cdata',...
                                cell(1,number_of_samples), ...
                                'colormap', ...
                                cell(1,number_of_samples) ...
                                );

            for n = 1:number_of_samples

                if(self.video_enabled)
                    self.video(n) = hyperplane_frame(self.W,DATA);
                end

                self = self.partial_fit(X(:,n),Y(:,n));

            end

        end

        % Need to be implemented for any ArxModel
        function self = calculate_output(self,x)
            self.yh = self.W' * x;
        end

        % Need to be implemented for any ArxModel
        function number_of_outputs = get_number_of_outputs(self)
            [number_of_outputs,~] = size(self.W);
        end        

    end
    
end