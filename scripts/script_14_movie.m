%% ------------------ MOVIE TEST 1

clear; close all;

timestep = 1;
fr = 1;

x = 0;
y = 0;

vx = 1;
vy = 1;

for t = 0:timestep:10,
    
    cla; % clear all axes properties and deletes all objects
    plot(x,y,'b.','MarkerSize',20)
    axis([0 10 0 10])
    %drawnow
    M1(fr) = getframe;
    
    x = x + vx*timestep;
    y = y + vy*timestep;
    fr = fr + 1;
    
end

movie(M1,2)  % play 2 times forward
movie(M1,-1) % negative: play 1 time forward, and 1 time backward

%% ------------------ MOVIE TEST 2

clear; close all;

% Creates a 2D Mesh to plot surface
x = linspace(0,1,100);
[X,Y] = meshgrid(x,x);

N=100; % Number of frames
for i = 1:N
    
    Z = sin(2*pi*(X-i/N)).*sin(2*pi*(Y-i/N));
    
    cla; % clear all axes properties and deletes all objects
    surf(X,Y,Z)

    % Store the frame
    M2(i)=getframe; % leaving gcf out crops the frame in the movie.
end

movie(M2)

%% ------------------ MOVIE TEST 3

vidObj = VideoWriter('peaks.avi','Uncompressed AVI');
myVideo.FrameRate = 02;  % Default 30
myVideo.Quality = 50;    % Default 75
open(vidObj);

% Create an animation.
Z = peaks; surf(Z);
axis tight
set(gca,'nextplot','replacechildren');

for k = 1:20
   surf(sin(2*pi*k/20)*Z,Z)

   % Write each frame to the file.
   currFrame = getframe;
   writeVideo(vidObj,currFrame);
end
  
% Close the file.
close(vidObj);

%% ------------------ MOVIE TEST 4

readerobj = VideoReader('peaks.avi', 'tag', 'myreader1');
 
       % Read in all video frames.
       vidFrames = read(readerobj);
 
       % Get the number of frames.
       numFrames = get(readerobj, 'NumberOfFrames');
 
       % Create a MATLAB movie struct from the video frames.
       for k = 1 : numFrames
             mov(k).cdata = vidFrames(:,:,:,k);
             mov(k).colormap = [];
       end
 
       % Create a figure
       hf = figure; 
       
       % Resize figure based on the video's width and height
       set(hf, 'position', [150 150 readerobj.Width readerobj.Height])
 
       % Playback movie once at the video's frame rate
       movie(hf, mov, 1, readerobj.FrameRate);
	   
%% ------------------ MOVIE TEST 5

% A = PAR_acc{1,1}.VID;
% v = VideoWriter('newfile.avi');
% v.FrameRate = 10;
% open(v);
% writeVideo(v,A);
% close(v);

%% ------------------ MOVIE TEST X
