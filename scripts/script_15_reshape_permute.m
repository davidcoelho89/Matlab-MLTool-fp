%% ------------------ RESHAPE AND PERMUTE MATRIX

A = [1 2 3 ; 4 5 6 ; 7 8 9];
B = [10 11 12 ; 13 14 15 ; 16 17 18];
C(:,:,1) = A;
C(:,:,2) = B;
D = permute(C,[2 3 1]);
E = permute(D,[3 1 2]);
F = reshape(E,[],size(E,3),1);
G = F';
H = C;
H(:,:,3) = [19 20 21; 22 23 24; 25 26 27];
I = reshape(H,[],size(H,3),1);
J = reshape(H,[],size(H,3));

d = squeeze(c(1,:,:));  %transforma de 2x2x2 para 2x2
                        % "comprime 1 dimensão da matriz"
                        
%% END