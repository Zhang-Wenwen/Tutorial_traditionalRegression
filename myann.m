%��ȡѵ������
clear
clc
%------���������ANN�ԾƵ�������з���
%f1 f2 ��2������ֵ
% load data_KLDA.mat
% data=data_KLDA;
% load data_PCA.mat
% data=data_PCA;
load data_LDA.mat;
data=data_LDA;
% load data_KPCA.mat
% data=data_KPCA;
[P_m,~]=size(data);
randIndex = randperm(size(data,1));
data=data(randIndex,:); 
f2= data(1:floor(P_m*0.8),3);
f1=data(1:floor(P_m*0.8),2);
class=data(1:floor(P_m*0.8),1);
%����ֵ��һ��
[input,minI,maxI] = premnmx( [f1 , f2]')  ;
%�����������
s = length( class) ;
output = zeros( s , 3  ) ;
for i = 1 : s 
   output( i , class( i )  ) = 1 ;
end
%����������
net = newff( minmax(input) , [10 3] , { 'logsig' 'purelin' } , 'traingdx' ) ; 
%{
    minmax(input)����ȡ4�������źţ��洢��f1 f2�У������ֵ����Сֵ��
    [10,3]����ʾʹ��2�����磬��һ������ڵ���Ϊ10���ڶ�������ڵ���Ϊ3��
    { 'logsig' 'purelin' }��
        ��ʾÿһ����Ӧ��Ԫ�ļ������
        ������һ����Ԫ�ļ����Ϊlogsig�����Ժ��������ڶ���Ϊpurelin������S��ת�ƺ�����
    'traingdx'����ʾѧϰ������õ�ѧϰ����Ϊtraingdx���ݶ��½�����Ӧѧϰ��ѵ��������
%}
%����ѵ������
net.trainparam.show = 50 ;% ��ʾ�м���������
net.trainparam.epochs = 500 ;%	��ѧϰ������
net.trainparam.goal = 0.01 ;%������ѵ����Ŀ�����
net.trainParam.lr = 0.01 ;%ѧϰ���ʣ�Learning rate��
%��ʼѵ��
%����inputΪѵ�����������źţ���ӦoutputΪѵ������������
net = train( net, input , output' ) ;
%================================ѵ�����====================================%
%=============================���������в���=================================%
 
%��ȡ��������
% [t1 t2 c] = textread('testData.txt' , '%f%f%f%f%f',150);
 t1=data(floor(P_m*0.8)+1:end,2);
 t2=data(floor(P_m*0.8)+1:end,3);
 c=data(floor(P_m*0.8)+1:end,1);
%�������ݹ�һ��
testInput = tramnmx ( [t1,t2]' , minI, maxI ) ;
%%
% testInput_1=[testInput;c'];
% p=net(testInput_1);
%[testInput,minI,maxI] = premnmx( [t1 , t2 ]')  ;
%����
%����netΪѵ����õ������磬���ص�YΪ
Y = sim( net , testInput )
% p=net(testInput);
%ͳ��ʶ����ȷ��
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
indexes=zeros(s2,1);  %���ڼ�¼���Ԥ���Ƶ�����Ľ��
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;
    indexes(i)=Index;
    if( Index  == c(i)   ) 
        hitNum = hitNum + 1 ; 
    end
end
sprintf('ʶ������ %3.3f%%',100 * hitNum / s2 )

%% VI. ��ͼ
figure
plot(1:length(indexes),indexes,'r-*')
hold on
plot(1:length(indexes),c,'b:o')
grid on
legend('��ʵ���','Ԥ�����')
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ���������Խ���Ա�';
          ['accuracy = ' num2str(100 * hitNum / s2) '%']};
title(string)