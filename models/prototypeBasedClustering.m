classdef prototypeBasedClustering
    %
    % --- Common features for all Prototype-Based Classifiers ---
    %
    % Properties (Hyperparameters)
    %  
    %   - distance_measure = which measure used to compare two vectors
    %      (used in a lot of functions for prototype-based classifiers)
    %   - kernel_type = which kernel used (kernel based classifiers)
    %       = 'none' (it is not a kernel based model)
    %
    % Properties (Parameters)
    %
    %   - Cx = Clusters' centroids (prototypes)
    %	- winners = The closest prototype for each sample [1 x N]
    %   - winner = Closest prototypes to a sample [1 x 1]
    %	- distances = Distance from prototypes to each sample [Nk x N]
    %   - distance = Distance from prototypes a sample [Nk x 1]
    %
    % Methods
    %
    %   - prototypeBasedClustering()    % Constructor
    %   - predict(self,X)               % Prediction Function
    %
    % ----------------------------------------------------------------

    % Hyperparameters
    properties
        distance_measure = 2;
        kernel_type = 'none';
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Cx = [];              % Clusters' centroids (prototypes)
        winners = [];         % Closest prototypes to each sample  [1 x N]
        winner = [];          % Closest prototypes to a sample     [1 x 1]
        distances = [];       % Distance of prot to each sample    [Nk x N]
        distance = [];        % Distance of prot to a sample       [Nk x 1]
    end

    methods

        % Constructor
        function self = prototypeBasedClustering()
            % Set the hyperparameters after initializing!
        end

        % Training Function (1 instance)
        % function self = partial_fit(self,x,y)
        %     % Each classifier has it own function
        % end

        % Training Function (N instances)
        % function self = fit(self,X,Y)
        %   "Each classifier has it own function"
        % end
        
    end % end methods

    methods (Static)
        
        function winner = findWinnerPrototype(Cx,sample,self)

            [~,Nk] = size(Cx);
            Vdist = zeros(1,Nk);
            
            for i = 1:Nk
                Vdist(i) = vectorsDistance(Cx(:,i),sample,self);
            end

            if(self.distance == 0) % dot product
                [~,winner] = max(Vdist);
            else % other distance measures
                [~,winner] = min(Vdist);
            end

        end

    end

end % end class