%% Init

clear;
clc;
close;

%% Get strings from txt

fid = fopen('PCoSA.txt');
txt_PCoSA = textscan(fid,'%s','delimiter','\n');
fclose(fid);
clear fid;

%% Define patterns to get

[nlines,~] = size(txt_PCoSA{1,1});
npatterns = nlines/4;

input_size = length(txt_PCoSA{1,1}{3,1});
output_size = length(txt_PCoSA{1,1}{4,1}) - 2;

input_matrix = zeros(npatterns,input_size);
output_matrix = zeros(npatterns,output_size);

for i = 1:npatterns
    % fill input matrix
    for j = 1:input_size
        input_matrix(i,j) = str2double(txt_PCoSA{1,1}{4*i-1,1}(j));
    end

    % fill output matrix
    for j = 1:output_size
        output_matrix(i,j) = str2double(txt_PCoSA{1,1}{4*i,1}(j));
    end
end

%% Clear and Variables

clear i j ans

save('ecc.mat','input_matrix','output_matrix');


%% end











