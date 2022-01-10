function [acc,theta1,L,output] = logisticRegression(train_matrix,train_label,test_matrix,test_label,Features,output)
%logisticRegression
%train_matrix,train_label,test_matrix,test_label,Features�ֱ���ѵ���������Լ������������ĸ���
%   ����Ԥ����ȷ�ʣ���������Ӧ����ͼ��
%% ��ʼѵ��
%�趨ѧϰ��Ϊ0.01
delta=1;
lamda=0.3; %������ϵ��
[m1,~]=size(train_matrix);
[m2,~]=size(test_matrix);
theta1=rand(1,Features);
%theta1=[.5,.5];
%%ѵ��ģ��

%�ݶ��½��㷨���theta��ÿ�ζ��Ƕ�ȫ�������ݽ���ѵ����
num = 700; %����������
L=[];
while(num)
    dt=zeros(1,Features);
    loss=0;
    for i=1:m1
        xx=train_matrix(i,1:Features);
        yy=train_label(i,1);
        h=1/(1+exp(-(theta1 * xx')));
        dt=dt+(h-yy) * xx;
        loss=loss+ yy*log(h)+(1-yy)*log(1-h);
    end
    loss=-loss/m1;
    L=[L,loss];
    
    theta2=theta1 - delta*dt/m1 - lamda*theta1/m1;
    theta1=theta2;
    num = num - 1;
    
    if loss<0.01
        break;
    end
end


%%
%��������
acc=0;

for i=1:m2
    xx=test_matrix(i,1:Features)';
    yy=test_label(i);
    finil=1/(1+exp(-theta2 * xx));
    if finil>0.5 && yy==1
        acc=acc+1;
        output(i)=1;
    elseif finil<=0.5 && yy==0
        acc=acc+1;
    end
end
disp('����Ԥ��ĸ���Ϊ')
disp(acc/m2)
end

