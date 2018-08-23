## 课时35  基本操作  13:59

```matlab
>> 5+6

ans =

    11

>> 3-2

ans =

     1

>> 5*8

ans =

    40

>> 1/2

ans =

    0.5000

>> 2^6

ans =

    64

>> 1 == 2 %false

ans =

     0

>> 1 ~= 2

ans =

     1

>> 1 && 0  %AND

ans =

     0

>> 1 || 0  %OR

ans =

     1

>> xor(1,0)

ans =

     1

>> 
>> 
>> a = 3

a =

     3

>> b = 3; %semiconlon supressing output
>> b

b =

     3

>> c = 'hi'

c =

hi

>> d = (3 >= 1)

d =

     1

>> e = pi;
>> e

e =

    3.1416

>> disp(e)
    3.1416

>> disp(sprintf('2 decimals: %0.2f',a))
2 decimals: 3.00
>> disp(sprintf('2 decimals: %0.2f',e))
2 decimals: 3.14
>> 
>> 
>> A = [1 2; 3 4; 5 6;]

A =

     1     2
     3     4
     5     6

>> B = [1 2;
3 4;
5 6]

B =

     1     2
     3     4
     5     6

>> v = [1 2 3]

v =

     1     2     3

>> u = [1;2;3]

u =

     1
     2
     3

>> v = 1:0.1:2

v =

  1 至 7 列

    1.0000    1.1000    1.2000    1.3000    1.4000    1.5000    1.6000

  8 至 11 列

    1.7000    1.8000    1.9000    2.0000

>> v = 1:6

v =

     1     2     3     4     5     6

>> ones(2,3)

ans =

     1     1     1
     1     1     1

>> f = 2*ones(2,3)

f =

     2     2     2
     2     2     2

>> g = [2 2 2; 2 2 2; 2 2 2]

g =

     2     2     2
     2     2     2
     2     2     2

>> h = ones(1,3)

h =

     1     1     1

>> i = zeros(1,3)

i =

     0     0     0

>> w = rand(1,3)

w =

    0.8147    0.9058    0.1270

>> rand(3,3)

ans =

    0.9134    0.2785    0.9649
    0.6324    0.5469    0.1576
    0.0975    0.9575    0.9706

>> rand(3,3)

ans =

    0.9572    0.1419    0.7922
    0.4854    0.4218    0.9595
    0.8003    0.9157    0.6557

>> w = randn(1,3)

w =

   -1.2075    0.7172    1.6302

>> w = randn(1,3)

w =

    0.4889    1.0347    0.7269

>> w = -6 + sqrt(10)*(randn(1,1000));

>> hist(w)
>> hist(w,50)
>> eye(4)

ans =

     1     0     0     0
     0     1     0     0
     0     0     1     0
     0     0     0     1
```


## 课时36  移动数据  16:07

>> A = [1 2; 3 4; 5 6]

A =

     1     2
     3     4
     5     6

>> size(A)  %return the size of the matrix

ans =

     3     2

>> sz = size(A)

sz =

     3     2

>> sz % now sz is a matrix with the first element 3 and the second element is 2

sz =

     3     2

>> size(sz)

ans =

     1     2

>> size(A,1) %return the size of the first dimension of A

ans =

     3

>> size(A,2)

ans =

     2

>> v = [1 2 3 4]

v =

     1     2     3     4

>> length(v) % the size of the longest dimension

ans =

     4

>> length(A) %the longer size is 3

ans =

     3

>> length([1;2;3;4;5])

ans =

     5

>> 
>> 
>> 
>> % load data and find data on the file system
>> 
>> pwd

ans =

C:\Users\谢昀臻\Documents\MATLAB

>> cd 'C:'\!WorkSpace\'
 cd 'C:'\!WorkSpace\'
    ↑
错误: 字符串未正常终止。
 
>> cd 'C:\!WorkSpace\'
>> ls

.                    Arduino              JAVA                 ebooks               test.ino             
..                   C                    Python               eeee                 
AndrewNg-ML-note-md  C++                  Python-Crawler       scrapy               

>> % using " load 'filename' " command to load data file
>> % who command can show the whole variables I have in my workspace
>> 
>> who

您的变量为:

A    ans  sz   v    

>> % "whos" command gives you the detailed view 
>> whos
  Name      Size            Bytes  Class     Attributes

  A         3x2                48  double              
  ans       1x29               58  char                
  sz        1x2                16  double              
  v         1x4                32  double              

>> % using "clear __" to clear some variables
>> % let priceV a 47-d vector
>> priceV = [1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;47]

priceV =

     1
     2
     3
     4
     5
     6
     7
     8
     9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47

>> % give 10 elements of vector Y  to the v
>> v = priceY(1:10)
未定义函数或变量 'priceY'。
 
是不是想输入:
>> v = priceV(1:10)

v =

     1
     2
     3
     4
     5
     6
     7
     8
     9
    10

>> % save the v as a file called hello.mat
>> save hello.mat v;
>> clear 
>> whos 
>> who
>> load hello.mat
>> who

您的变量为:

v  

>> whos
  Name       Size            Bytes  Class     Attributes

  v         10x1                80  double              

>> v

v =

     1
     2
     3
     4
     5
     6
     7
     8
     9
    10

>> save hello.txt v -ascii  & save as text(ASCII)
错误使用 save
'&' 不是有效的变量名称。
 
>> save hello.txt v -ascii  % save as text(ASCII)
>> A = [1 2; 3 4; 5 6;]

A =

     1     2
     3     4
     5     6

>> A(3,2) %index

ans =

     6

>> A(2,:)   % fetch everything in the second row

ans =

     3     4

>> % ":" means every element along that row/column
>> A(:,2)

ans =

     2
     4
     6

>> A([1 3],:)

ans =

     1     2
     5     6

>> 
>> 
>> A

A =

     1     2
     3     4
     5     6

>> A(:,2) = [10; 11; 12]

A =

     1    10
     3    11
     5    12

>> A = [A,[100;101;102]];    % append another column vector to right
>> A

A =

     1    10   100
     3    11   101
     5    12   102

>> A(:)    % put all elements of A into a single vector

ans =

     1
     3
     5
    10
    11
    12
   100
   101
   102

>> A = [1 2; 3 4; 5 6];
>> B = [11 12; 13 14; 15 16]

B =

    11    12
    13    14
    15    16

>> C = [A B]    % concatenating onto each other

C =

     1     2    11    12
     3     4    13    14
     5     6    15    16

>> D = [A; B]    % put the next thing at the bottom

D =

     1     2
     3     4
     5     6
    11    12
    13    14
    15    16

>> [A B]

ans =

     1     2    11    12
     3     4    13    14
     5     6    15    16

>> [A, B]

ans =

     1     2    11    12
     3     4    13    14
     5     6    15    16

>> 

## 课时37  计算数据  13:15
## 课时38  数据绘制  09:38
## 课时39  控制语句：for，while，if 语句 12:56
## 课时40  矢量  13:48
## 课时41  本章课程总结