clear all
clc

Dataname = 'bbcsport4vbigRnSp'
percentDel = 0.3
Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datafold);

lambda1 = 0.1 
lambda2 = 0.0001 
lambda3 = 300 

f = 1
r = 2
ind_folds = folds{f};

truthF = truth;   % ��ʵ���
clear truth
numInst = length(truthF);
num_cluster = length(unique(truthF));
for iv = 1:length(X)
    X1 = X{iv};     % bbc����������������� ת�úͲ�ת��
    X1 = NormalizeFea(X1,0);    % 0Ϊ�й�һ��
    ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
    X1(:,ind_0) = [];           % ȥ��ȱʧ�ӽ�  
    % -------------- ͼ��ʼ�� ----------------- %
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 11;
    options.WeightMode = 'Binary';       % Binary
    So{iv} = constructW(X1',options);                  % ��������� ͼ��ʼ������    
    G = diag(ind_folds(:,iv));
    G(:,ind_0) = [];
    Sor{iv} = G*So{iv}*G';
end
clear X X1 ind_0 G So

max_iter = 50;
S_ini = zeros(numInst,numInst);
for iv = 1:length(Sor)
    S_ini = S_ini + Sor{iv};
end

[pre_labels] = OGIMC(Sor,S_ini,num_cluster,numInst,lambda1,lambda2,lambda3,max_iter,ind_folds)

result_cluster = ClusteringMeasure(truthF, pre_labels)*100  




 