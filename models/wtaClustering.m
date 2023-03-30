classdef wtaClustering
    %
    % --- Winner Takes-All Clustering Function ---
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
    %   - wtaClustering()       % Constructor
    %   - partial_fit(self,x,y) % fit Function (1 instance)
    %   - fit(self,X,Y)         % fit Function (N instances)
    %   - predict(self,X)       % Prediction Function
    %
    % ----------------------------------------------------------------

    % Hyperparameters
    properties
        distance_measure = 2;
        kernel_type = 'none';
        number_of_epochs = 200;
        number_of_prototypes = 20;
        initialization_type = 2;
        learning_type = 2;
        learning_step_initial = 0.7;
        learning_step_final = 0.01;
        video_enabled = 0;
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Cx = [];              % Clusters' centroids (prototypes)
        winners = [];         % Closest prototypes to each sample [1 x N]
        winner = [];          % Closest prototypes to a sample    [1 x 1]
        distances = [];       % Distance of prot to each sample   [Nk x N]
        distance = [];        % Distance of prot to a sample      [Nk x 1]
        video_structure = []; % played by 'video function'        [1 x Nep]
    end

    methods

        % Constructor
        function self = wtaClustering()
            % Set the hyperparameters after initializing!
        end

        % Training Function (1 instance)
        function self = partial_fit(self,x,y)
            % ToDo - all
        end

        % Training Function (N instances)
        function self = fit(self,X,Y)
            % ToDo - all
        end
        
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