clear;
clc;
% load normal.mat
X=xlsread("wine.csv"); % X训练数据集
% X = X()
[Xrow, Xcol] = size(X); % Xrow：样本个数 Xcol：样本属性个数
%% 数据预处理，进行标准化出理，处理后均值为0方差为1
Xc = mean(X); % 求原始数据的均值
Xe = std(X); % 求原始数据的方差
X0 = (X-ones(Xrow,1)*Xc) ./ (ones(Xrow,1)*Xe); % 标准阵X0,标准化为均值0，方差1;
c = 20000; %此参数可调
%% 求核矩阵
for i = 1 : Xrow
for j = 1 : Xrow
K(i,j) = exp(-(norm(X0(i,:) - X0(j,:)))^2/c);%求核矩阵，采用径向基核函数，参数c
end
end
%% 中心化矩阵
n1 = ones(Xrow, Xrow);
N1 = (1/Xrow) * n1;
Kp = K - N1*K - K*N1 + N1*K*N1; % 中心化矩阵
%% 特征值分解
[V, D] = eig(Kp); % 求协方差矩阵的特征向量（V）和特征值（D）
lmda = real(diag(D)); % 将主对角线上为特征值的对角阵变换成特征值列向量
[Yt, index] = sort(lmda, 'descend'); % 特征值按降序排列，t是排列后的数组，index是序号
%% 确定主元贡献率 记下累计贡献率大于85%的特征值的序号放入 mianD中
rate = Yt / sum(Yt); % 计算各特征值的贡献率
sumrate = 0; % 累计贡献率
mpIndex = []; % 记录主元所在特征值向量中的序号
for k = 1 : length(Yt) % 特征值个数
sumrate = sumrate + rate(k); % 计算累计贡献率
mpIndex(k) = index(k); % 保存主元序号
if sumrate > 0.4
break;
end
end
npc = length(mpIndex); % 主元个数
%% 计算负荷向量
for i = 1 : npc
zhuyuan_vector(i) = lmda(mpIndex(i)); % 主元向量
P(:, i) = V(:, mpIndex(i)); % 主元所对应的特征向量（负荷向量）
end
zhuyuan_vector2 = diag(zhuyuan_vector); % 构建主元对角阵

%% 

[m,~]=size(X);
N1= X(X(:,1)==1,:);
N2= X(X(:,1)==2,:);
N3= X((X(:,1)==3),:);
[m1,~]=size(N1);
[m2,~]=size(N2);
[m3,~]=size(N3);
if npc==3
C=[ones(m1,1);2*ones(m2,1);3*ones(m3,1)];
figure 
scatter3(P(:,1),P(:,2),P(:,3),20,C,'filled');
title('KPCA-3-dimension')
elseif npc==2
    %% 
    figure
plot(P(1:m1,1),P(1:m1,2),'r*')
hold on 
plot(P(m1+1:m2+m1,1),P(m1+1:m2+m1,2),'go')
hold on 
plot(P(m2+m1+1:m2+m1+m3,1),P(m1+m2+1:m1+m2+m3,2),'bs')
title('KPCA-2-dimension,PolyPlus')
else
    disp('please redrow')
end
%% 保存降维后数据并分为训练集和测试集
data_KPCA=[X(:,1) P];
save ('data_KPCA.mat','data_KPCA');