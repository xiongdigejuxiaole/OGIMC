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
    %���Լ���ת�þ���ÿ��Ԫ��ȡ�������
end;

[v,d] = eig(A);
%�����A��ȫ������ֵ�����ɶԽ���d������������v��v��������Ӧ����������

d = diag(d);
%d = real(d);
%����ֵ�ӶԽ�ȫ���Ƶ���һ�У�d����о���

if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
    %��������
end;

%d1Ϊ�������о���
%idxΪ�����d��Ӧ���±���о���

idx1 = idx(1:c);
%��ȡǰc���±�

eigval = d(idx1);
%��idx��d������ֵ����ȡ����

eigvec = v(:,idx1);
%��idx��v(��������)��ȡ����

eigval_full = d(idx);
%ȫ������ֵ