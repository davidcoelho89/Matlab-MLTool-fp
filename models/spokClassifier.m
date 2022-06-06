classdef spokClassifier
   % HELP about spokClassifier
   
   % Hyperparameters
   properties
        number_of_epochs = 5;
        is_stationary = 0;
        design_method = 'per_class';
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
       video = []; % frame structure (can be played with 'video function')
       Yh = [];    % Hold all predictions
   end
   
   methods
       
       % Constructor
       function self = spokClassifier()
           % Set the hyperparameters after initializing!
       end
       
       % Training Function (1 instance)
       function self = partial_fit(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       % Training Function (N instances)
       function self = fit(self,X,Y)
           % ToDo - All
           self = self + X + Y;
       end
       
       % Prediction Function
       function Yh = predict(self,X)
           % ToDo - All
           Yh = prototypesClassify(self,X);
       end
       
       function self = dictionaryGrow(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       function self = dictionaryUpdate(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
           
       
   end    
    
end








