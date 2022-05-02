classdef movingBlip < blip
    % teste
    
    properties
        deltaAoA
    end
    
    methods
        
        function self = movingBlip(deltaAoA, varargin)
            % Assign superclass
            self = self@blip(varargin{:});
            % Assign unique property
            if nargin >= 1
                self.deltaAoA = deltaAoA;
            end
        end
        
        function self = move(self)
            self.AoA = self.AoA + self.deltaAoA;
            if abs(self.AoA) > 65
                self.deltaAoA = -self.deltaAoa;
            end
        end
        
    end
    
end