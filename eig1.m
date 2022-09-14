function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
    %和自己的转置矩阵，每个元素取二者最大
end;

[v,d] = eig(A);
%求矩阵A的全部特征值，构成对角阵d，并产生矩阵v，v各列是相应的特征向量

d = diag(d);
%d = real(d);
%特征值从对角全部移到第一列，d变成列矩阵

if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
    %倒序排序
end;

%d1为排序后的列矩阵
%idx为排序后d对应的下标的列矩阵

idx1 = idx(1:c);
%截取前c个下标

eigval = d(idx1);
%按idx把d（特征值）截取出来

eigvec = v(:,idx1);
%按idx把v(特征向量)截取出来

eigval_full = d(idx);
%全部特征值