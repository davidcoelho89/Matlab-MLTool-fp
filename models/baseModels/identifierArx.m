classdef identifierArx
    %
    % --- Common features for all Arx Identifiers ---
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
    %   - self = identifierArx() 
    %   - self = predict(self,X) 
    %   - self = partial_predict(self,x)
    %   - self = update_memory_from_prediction(self)
    %   - xn_out = update_regression_vector_from_memory(self,x)
    %
    % ----------------------------------------------------------------

    % Hyperparameters
    properties
        
        add_bias = 1;           % Add bias as default
        prediction_type = 1;    % 1-step ahead
        output_lags = 2;        % considering just one output with 2 lags

    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)

        Yh = [];
        yh = [];
        last_predictions_memory = [];

    end

    methods

        % Constructor
        function self = identifierArx()
            % Set the hyperparameters after initializing
        end

        % Prediction function (1 instance)
        function self = partial_predict(self,x)

            if(self.add_bias == 1)
                x = [1 ; x];
            end

            x = self.update_regression_vector_from_memory(x);
            self = self.calculate_output(x);
            self = self.update_memory_from_prediction();

        end

        % Prediction Function (N instances)
        function self = predict(self,X)
            
            [~,number_of_samples] = size(X);
            number_of_outputs = get_number_of_outputs(self);
            
            self.Yh = zeros(number_of_outputs,number_of_samples);
            
            self.last_predictions_memory = X(1:sum(self.output_lags),1);
            
            for n = 1:number_of_samples
                self = self.partial_predict(X(:,n));
                self.Yh(:,n) = self.yh;
            end
            
        end

        function self = update_memory_from_prediction(self)
            
            updated_memory = zeros(length(self.last_predictions_memory),1);
            
            initial_sample_index = 1;
            for i = 1:length(self.output_lags)
                % Get lag and last sample for present output
                lag = self.output_lags(i);
                final_sample_index = initial_sample_index + lag - 1;
                % Update memory from specific output
                if(lag == 1)
                    updated_memory(initial_sample_index) = self.yh(i);
                else
                    updated_memory(initial_sample_index:final_sample_index,1) = ...
                    [self.yh(i); ...
                     self.last_predictions_memory(initial_sample_index:final_sample_index-1,1)];
                end
                % Update Initial sample for next output
                initial_sample_index = final_sample_index + 1;
            end

            self.last_predictions_memory = updated_memory;
            
        end
        
        function self = hold_last_output_from_fit(self,y)
            self.yh = y;
            self = self.update_memory_from_prediction();
        end
        
        function xn_out = update_regression_vector_from_memory(self,x)

            xn_out = x;
            
            if(self.prediction_type == 0)       % free simulation
                if(self.add_bias)
                    xn_out(2:length(self.last_predictions_memory)+1,1) = self.last_predictions_memory;
                else
                    xn_out(1:length(self.last_predictions_memory),1) = self.last_predictions_memory;
                end
            elseif(self.prediction_type == 1)   % 1-step ahead
                % Does nothing
            else                                % n-steps ahead
                % ToDo - Verify
            end

        end        

    end
    
end % end class