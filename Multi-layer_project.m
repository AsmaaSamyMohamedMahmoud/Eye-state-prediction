%%NN_Project (EyeState Prediction using neural network)
%%Mustafa Bal - Asmaa Samy 
%%using Multi-Layer Perceptron - Online Learning


clear
clc
close all
%Loading the training dataset (80%)

load EEG_Training80.txt;
Xa = EEG_Training80 (:, [1:14]);
X=Xa*0;
for xxx =1:1:14
X(:,xxx)= (Xa(:,xxx) - mean(Xa(:,xxx)))/(std(Xa(:,xxx))); %%Normalization
end
X = [ones(size (Xa,1),1) X]';  %%Input layer includes (14 features + bias term)
T = EEG_Training80(:, 15);  %% True Output 

%Loading the validation dataset (20%)

load EEG_Validation20.txt;
X_validate = EEG_Validation20(:, [1:14]);

Xn=X_validate*0;
for ii =1:1:14
Xn(:,ii)= (X_validate(:,ii) - mean(X_validate(:,ii)))/(std(X_validate(:,ii)));
end
Xn = [ones(size (X_validate,1),1) Xn]'; 
Tn = EEG_Validation20(:, 15);

P= 1;    %No of output units 
D = 15;  %No of input units 
H = 20;  %No of hidden units 
K = P;
delta_w =zeros(D,H);
delta_v =zeros(H,P);

%Initiliazed weights 
w = rand(D,H);        % 'w' is from input to hidden unit
w = (w-0.5)*2*0.01;
v = rand(H,P);        % 'v' is from hiden to output unit
v = (v-0.5)*2*0.01;   %Initializing the weights randomly from -0.01 to 0.01

lr = 0.1;  % learning rate
num_of_instances = size (X,2);
testinstance=size(Xn,2);
num_of_epoch = 300;
error_training=0;
error_validation=0;

%Training part

for e=1:1:num_of_epoch
    for instance=1:1:num_of_instances
        o1 = (w')*X(:,instance);      
        z = 1./(1+exp(-o1)); %Sigmoid for activatin on hidden layer
        o2 = (v')*z;
        y(:,instance)=1./(1+exp(-o2)); %Predicted Output
        
%Weights Updating Part
        for i=1:1:P
            for h=1:1:H
              delta_v(h,i) = 0;
            end
        end
        for i=1:1:P %For 'v' hidden to output
            for h=1:1:H        
              delta_v(h,i) = (T(instance,i)-y(:,instance))*z(h); 
            end
        end        
        for h = 1:1:H %For 'w' input to hidden
            for j=1:1:D
                 delta_w(j,h) = 0;
                for i = 1:1:P
                    delta_w(j,h) = delta_w(j,h)+ (T(instance,i)-y(:,instance))*v(h,i);
                end
                delta_w(j,h) = delta_w(j,h)*z(h)*(1-z(h))*X(j,instance);
            end
        end
        for i = 1:1:P
            for h=1:1:H
                v(h,i) = v(h,i) + lr*delta_v(h,i);
            end
        end
        for h = 1:1:H
            for j=1:1:D
                w(j,h) = w(j,h) + lr*delta_w(j,h);
            end
        end        
    end
    weights_v(:,:,e) = v;
    weights_w(:,:,e) = w;
    num_err_out = 0;
    t1 = threshold(y); %Thresholding function
    t2=double(t1); %Doubled to get confusion matrix
    error_training=1/num_of_instances*((sum((T-t2').^2))); %Error function
    plot_error_train(e)=error_training;
    %fprintf('%d \n',error_training);
    
  %% Validation Part
    o_test = (w)'*Xn; 
    z_test=1./(1+exp(-o_test));
    o_test1 = (v')*z_test;
    ytest = 1./(1+exp(-o_test1));
    t_test = thresholdtesting(ytest);
    t_test2=double(t_test);
    error_validation=1/testinstance*((sum((Tn-t_test2').^2)));
    plot_error_validate(e)=error_validation;

end
%%calculating the accuracy from the confusion matrix 
C=confusionmat(T,t2);
train_acc = (C(1,1)+ C(2,2)) / num_of_instances * 100
% %%Plotting the confusion matrix it will work in the latest version only 
% figure 
% confusionchart(C);
% title ('Confusion matrix for Training')
C1=confusionmat(Tn,t_test2);
val_acc = (C1(1,1)+ C1(2,2)) / testinstance * 100
% % figure 
% % confusionchart(C1);
% % title ('Confusion matrix for Validation')
% %%Plotting the training and validation errors through epochs 
%  plot(1:num_of_epoch,plot_error_train)
%  hold on 
%  plot(1:num_of_epoch,plot_error_validate)
%  hold off
% legend('training error','validation error')
% 
% figure
%%Plotting the change in synaptics weights (from input to hidden) through epochs
% subset_w=weights_w(:,[2 9 17],:);
% w_new = reshape(subset_w,[],300);  
% plot (1:num_of_epoch,w_new)
% ylabel('Synaptic weights "w"')
% xlabel ('No of Epochs')
% title('change in Synaptic weights through epochs')

% %%Plotting the change in synaptics weights (from hidden to output) through epochs
% v_new = reshape(weights_v,[],300);  
% plot (1:num_of_epoch,v_new)
% ylabel('Synaptic weights "v"')
% xlabel ('No of Epochs')
% title('change in Synaptic weights through epochs')