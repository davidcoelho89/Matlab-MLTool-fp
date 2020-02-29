%% Function Handle

% http://www.mathworks.com/help/matlab/matlab_prog/creating-a-function-handle.html

% can create function handles to named and anonymous functions.
% load and save them as you would any other variable.

% The function must be in the MATLAB path or be anonymous.
% Be careful with functions with the same name

% Use 1: Pass a function to another function
% Use 2: Specify callback functions
% Use 3: Construct handles to functions defined inline
% Use 4: Call local functions from outside the main function.

% Ex:

% function y = computeSquare(x)
% y = x.^2;
% end

% f = @computeSquare;
% a = 4;
% b = f(a) % b = 16

% Array or Structs of function handlers

C = {@sin, @cos, @tan};
C{2}(pi),

S.a = @sin;  S.b = @cos;  S.c = @tan;
S.a(pi/2),

% Information about the function handle

fh = @sin;
s = functions(fh),

%% Anonymous functions

% h = @(arglist) (anonymous_function)

% Without inputs
myfunction1 = @() datestr(now);
d1 = myfunction1(),   % get the result of function
d2 = myfunction1,     % get the function handle

% With multiple inputs
myfunction2 = @(x,y) (x^2 + y^2 + x*y);
x = 1;
y = 10;
z = myfunction2(x,y),

% With multiple outputs
c = 10;
myfunction3 = @(x,y) ndgrid((-x:x/c:x),(-y:y/c:y));
[x,y] = myfunction3(pi,2*pi);
z = sin(x) + cos(y);
mesh(x,y,z)

%% Pass a Function as parameter to another function 

a = 0;
b = 5;

q1 = integral(@log,a,b),
q2 = integral(@sin,a,b),
q3 = integral(@exp,a,b),

fun = @(x)x./(exp(x)-1);
q4 = integral(fun,0,Inf),

%% String to Function Handle X Function Handle to String

% when you have to call a function by its name

str = 'ones';
fh1 = str2func(str),
fh1(1,5),

str = '@(x)(7*x-13)';
fh2 = str2func(str),
fh2(3)

% When you have to display the name or format of function handle

fh = @(x,y)sqrt(x.^2+y.^2);
str = func2str(fh);
disp(['Anonymous function: ' str])

%% Advanced topics

% Parameterizing functions
% http://www.mathworks.com/help/matlab/math/parameterizing-functions.html

% Nested Functions
% http://www.mathworks.com/help/matlab/matlab_prog/nested-functions.html

%% END
