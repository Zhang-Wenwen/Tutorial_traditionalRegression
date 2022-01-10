clear;
clc;
% load normal.mat
X=xlsread("wine.csv"); % Xѵ�����ݼ�
% X = X()
[Xrow, Xcol] = size(X); % Xrow���������� Xcol���������Ը���
%% ����Ԥ�������б�׼������������ֵΪ0����Ϊ1
Xc = mean(X); % ��ԭʼ���ݵľ�ֵ
Xe = std(X); % ��ԭʼ���ݵķ���
X0 = (X-ones(Xrow,1)*Xc) ./ (ones(Xrow,1)*Xe); % ��׼��X0,��׼��Ϊ��ֵ0������1;
c = 20000; %�˲����ɵ�
%% ��˾���
for i = 1 : Xrow
for j = 1 : Xrow
K(i,j) = exp(-(norm(X0(i,:) - X0(j,:)))^2/c);%��˾��󣬲��þ�����˺���������c
end
end
%% ���Ļ�����
n1 = ones(Xrow, Xrow);
N1 = (1/Xrow) * n1;
Kp = K - N1*K - K*N1 + N1*K*N1; % ���Ļ�����
%% ����ֵ�ֽ�
[V, D] = eig(Kp); % ��Э������������������V��������ֵ��D��
lmda = real(diag(D)); % �����Խ�����Ϊ����ֵ�ĶԽ���任������ֵ������
[Yt, index] = sort(lmda, 'descend'); % ����ֵ���������У�t�����к�����飬index�����
%% ȷ����Ԫ������ �����ۼƹ����ʴ���85%������ֵ����ŷ��� mianD��
rate = Yt / sum(Yt); % ���������ֵ�Ĺ�����
sumrate = 0; % �ۼƹ�����
mpIndex = []; % ��¼��Ԫ��������ֵ�����е����
for k = 1 : length(Yt) % ����ֵ����
sumrate = sumrate + rate(k); % �����ۼƹ�����
mpIndex(k) = index(k); % ������Ԫ���
if sumrate > 0.4
break;
end
end
npc = length(mpIndex); % ��Ԫ����
%% ���㸺������
for i = 1 : npc
zhuyuan_vector(i) = lmda(mpIndex(i)); % ��Ԫ����
P(:, i) = V(:, mpIndex(i)); % ��Ԫ����Ӧ����������������������
end
zhuyuan_vector2 = diag(zhuyuan_vector); % ������Ԫ�Խ���

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
%% ���潵ά�����ݲ���Ϊѵ�����Ͳ��Լ�
data_KPCA=[X(:,1) P];
save ('data_KPCA.mat','data_KPCA');