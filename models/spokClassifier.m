classdef spokClassifier
    %
    % SParse Online adptive Kernel Classifier
    %
    % Properties (Hyperparameters):
    %
    %    number_of_epochs = <integer>
    %        if > 1, "shows the data" more than once to the algorithm
    %    is_stationary = [0 or 1]
    %
    %    design_method = how the dictionary will be built
    %        = 'one_dicitionary_per_class'
    %        = 'single_dictionary'
    %    sparsification_strategy = define how the prototypes will be selected
    %        = 'ald'
    %        = 'coherence'
    %        = 'novelty'
    %        = 'surprise'
    %    v1 =
    %    v2 =
    %    update_strategy = define how the prototypes will be updated
    %        = 'none'
    %        = 'wta' (lms, unsupervised)
    %        = 'lvq' (supervised)
    %    update_rate = <real>
    %        [0 to 1]
    %    pruning_strategy = define how the prototypes will be prunned
    %        = 'none'
    %        = 'drift_based'
    %        = 'hits_and_error'
    %    min_score =
    %    max_prototypes =
    %    min_prototypes =
    %    video_enabled =
    %    nearest_neighbors =
    %    knn_aproximation =
    %    kernel_type =
    %    regularization =
    %    sigma =
    %    alpha =
    %    theta =
    %    gamma =
    %
    % Properties (Parameters)
    %
    %    Cx = Clusters' centroids (prototypes)
    %    Cy = Clusters' labels
    %    Km = Kernel Matrix of Entire Dictionary
    %    Kinv = Kernel Matrix for each class (cell)
    %    Kinvc = Inverse Kernel Matrix for each class (cell)
    %    score = Used for prunning method
    %    classification_history = Used for prunning method
    %    times_selected = Used for prunning method
    %    video = frame structure (can be played with 'video function')
    %    Yh = Hold all predictions
    %
    % Methods:
    %
    %	spokClassifier()        % Constructor
    %	partial_fit(self,x,y)	% Training Function (1 instance)
    %	fit(self,X,Y)           % Training Function (N instances)
    %	Yh = predict(self,X)    % Prediction Function

    % Hyperparameters
    properties
        number_of_epochs = 1;
        is_stationary = 0;
        design_method = 'one_dicitionary_per_class';
        sparsification_strategy = 'ald';
        v1 = 0.1;
        v2 = 0.9;
        update_strategy = 'lms';
        update_rate = 0.1;
        pruning_strategy = 'error_score_based';
        min_score = -10;
        max_prototypes = 600;  % ("Budget")
        min_prototypes = 2;    % ("restriction")
        video_enabled = 0;
        nearest_neighbors = 1; % KNN
        knn_aproximation = 'majority_voting';
        kernel_type = 'gaussian';
        regularization = 0.001;
        sigma = 2;
        alpha = 1;
        theta = 1;
        gamma = 2;
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        Cx = [];     % Clusters' centroids (prototypes)
        Cy = [];     % Clusters' labels
        Km = [];     % Kernel Matrix of Entire Dictionary
        Kmc = [];    % Kernel Matrix for each class (cell)
        Kinv = [];   % Inverse Kernel Matrix of Dictionary
        Kinvc = [];  % Inverse Kernel Matrix for each class (cell)
        score = [];          % Used for prunning method
        class_history = [];  % Used for prunning method
        times_selected = []; % Used for prunning method
        video = [];  % frame structure (can be played with 'video function')
        Yh = [];           % all predictions (fit function)
        winners = [];      % closest prototype to each sample [1 x N]
        distances = [];    % distance of each sample from each prototype [Nk x N]
        near_indexes = []; % indexes for nearest prototypes [K x N]
        yh = [];           % last prediction (partial_fit function)
        winner = [];       % closest prototype to sample [1 x 1]
        distance = [];     % distance of sample from each prototype [Nk x 1]
        near_index = [];   % indexes for nearest prototypes [K x 1]


    end

    methods

        % Constructor
        function self = spokClassifier()
            % Set the hyperparameters after initializing!
        end

        % Training Function (1 instance)
        function self = partial_fit(self,x,y)

            number_of_classes = length(y);

            [~,number_of_prototypes_old] = size(self.Cx);

            if (number_of_prototypes_old == 0)
                % Make a guess (yh = [1 -1 -1 ... -1 -1]' : first class)
                self.yh = -1*ones(number_of_classes,1);
                self.yh(1) = 1;
                % Add sample to dictionary
                self = self.dictionaryGrow(x,y);
            else
                % Predict Output
                classification_output = prototypesClassify(self,x);
                self.yh = classification_output.Yh;
                self.winner = classification_output.winners;
                self.distance = classification_output.distances;
                self.near_index = classification_output.near_indexes;

                % Update number of times a prototype has been the winner
                self.times_selected(self.winner) = self.times_selected(self.winner) + 1;

                % Growing Strategy
                self = self.dictionaryGrow(x,y);

                % Update Strategy
                [~,number_of_prototypes_new] = size(self.Cx);
                if(number_of_prototypes_new == number_of_prototypes_old)
                    self = self.dictionaryUpdate(x,y);
                else
                    % For debug. Display dictionary size when it grows.
                    % disp(number_of_prototypes_new);
                end

                % Prunning Strategy
                self = self.updateScore(x,y);
                self = self.dictionaryPrune(x,y);

            end

        end

        % Training Function (N instances)
        function self = fit(self,X,Y)

            % Problem Initialization
            Xs = X;                  % Don't shuffle original data
            Ys = Y;                  % Don't shuffle original data
            [Nc,N] = size(Y);        % Total of classes and samples
            self.Yh = -1*ones(Nc,N); % Initialize outputs
            iteration = 0;           % 
            self.video = struct('cdata',cell(1,N*self.number_of_epochs),...
                                'colormap', cell(1,N*self.number_of_epochs));

            for epoch = 1:self.number_of_epochs

                for n = 1:N

                    if(self.video_enabled)
                        iteration = iteration + 1;
                        self.video(iteration) = prototypes_frame(self.Cx,X);
                    end

                    self = self.partial_fit(Xs(:,n),Ys(:,n));

                end

                % Shuffle Data
                if(self.is_stationary)
                    I = randperm(N);
                    Xs = Xs(:,I);
                    Ys = Ys(:,I);
                end

                % Hold last classification labels
                if(epoch == self.number_of_epochs)
                    self = self.predict(X,Y);
                end

            end % end epoch

        end % end fit

        % Prediction Function
        function self = predict(self,X)
            classification_output = prototypesClassify(self,X);
            self.Yh = classification_output.Yh;
            self.winners = classification_output.winners;
            self.distances = classification_output.distances;
            self.nearest_neighbors = classification_output.nearest_neighbors;
        end

        function self = dictionaryGrow(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

        function self = dictionaryUpdate(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

        function self = dictionaryPrune(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

        function self = addSample(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

        function self = removeSample(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

        function self = updateScore(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

    end

end