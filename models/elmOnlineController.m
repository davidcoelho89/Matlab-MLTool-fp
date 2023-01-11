classdef elmOnlineController
    
    % Hyperparameters
    properties
        openLoopSystem = [];
        forgiving_factor = 1;
        number_of_hidden_neurons = 25;
        non_linearity = 'sigmoid';
        number_of_epochs = 1;
        P_init = 10000;
        W_init = 0.01;
        
    end
    
	% Parameters
    properties (GetAccess = public, SetAccess = protected)
        W = [];
        P = [];
        structure = [];
        number_of_layers = [];
    end
    
    methods
        
        % Constructor
        function self = elmOnlineController(openLoopSystem)
            % Set other hyperparameters after initializing!
            self.openLoopSystem = openLoopSystem;
        end   
    
end













