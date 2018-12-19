clc, clear, close all;%, close all;
nr = 1;
seed = 1;
titels = {};

type = "LearningRate/";
%type = "Solver/";

if type == "LearningRate/" 
    titels  = {'0.2','0.4','0.6','0.8'};
elseif type == "Solver/"
    titels  = {'SGD','Adadelta','Adam'};
end



folder = '../results/'+type;


filenameInit = strcat(folder,titels{nr},'_seed',int2str(seed),'.txt');
dataInit = importdata(filenameInit);



epochs = size(dataInit);
epochs = epochs(1);
seeds = 4;
changed = size(titels);
changed = changed(2);



accA = zeros([epochs seeds]);
acc_valA = zeros([epochs seeds]);
lossA = zeros([epochs seeds]);
loss_valA = zeros([epochs seeds]);

acc_mean = zeros([epochs changed]);
acc_val_mean = zeros([epochs changed]);
loss_mean = zeros([epochs changed]);
loss_val_mean = zeros([epochs changed]);

for nr=1:changed
    for seed=1:seeds

        filename = strcat(folder,titels{nr},'_seed',int2str(seed),'.txt');

        data = importdata(filename);
        epoch = data(:,1);
        acc = data(:,2);
        acc_val = data(:,3);
        loss = data(:,4);
        loss_val = data(:,5);


        accA(:,seed)=acc;
        acc_valA(:,seed)=acc_val;
        lossA(:,seed)=loss;
        loss_valA(:,seed)=loss_val;

    end
    
    accAll{nr} = accA;
    acc_valAll{nr} = acc_valA;
    lossAll{nr} = lossA;
    loss_valAll{nr} = loss_valA;
%end

% -------------------------------------------------------------
% Val----------------------------------------------------------
% -------------------------------------------------------------

mean_accA = mean(accA')';
acc_mean(:,nr) = mean_accA;
std_accA = std(accA')';

mean_acc_valA = mean(acc_valA')';
acc_val_mean(:,nr) = mean_acc_valA;
std_acc_valA = std(acc_valA')';


mean_lossA = mean(lossA')';
loss_mean(:,nr) = mean_lossA;
std_lossA = std(lossA')';

mean_loss_valA = mean(loss_valA')';
loss_val_mean(:,nr) = mean_loss_valA;
std_loss_valA = std(loss_valA')';

%{
figure('Name',titels{nr});
errorbar(epoch,mean_accA,std_accA)
%plot(epoch,mean_accA)
xlim([0 epochs])
ylim([0 1])
hold on;
%plot(epoch,mean_acc_valA)
errorbar(epoch,mean_acc_valA,std_acc_valA)
title(titels{nr});
legend("acc","acc val",'Location','southeast')
xlabel('Epoch')
ylabel('Acc')


% -------------------------------------------------------------
% Loss---------------------------------------------------------
% -------------------------------------------------------------



figure('Name',titels{nr});
errorbar(epoch,mean_lossA,std_lossA)
xlim([0 epochs])
ylim([0 0.1])
hold on;
errorbar(epoch,mean_loss_valA,std_loss_valA)
title(titels{nr});
legend("loss","loss val",'Location','southeast')
ylabel('Loss')
xlabel('Epoch')

figure('Name',"std");
plot(epoch,std_lossA)
hold on
plot(epoch,std_loss_valA)
plot(epoch,std_accA)
plot(epoch,std_acc_valA)
title('std');
legend("std loss","std loss val","std acc","std acc val",'Location','northwest')
ylabel('std')
xlabel('Epoch')
% -------------------------------------------------------------
% Difference ----------------------------------------------------------
% -------------------------------------------------------------

diff_accA = mean_accA-mean_acc_valA;
diff_lossA = mean_loss_valA-mean_lossA;



figure('Name',titels{nr});
plot(epoch,diff_accA)
hold on;
plot(epoch,diff_lossA)
title(titels{nr});
legend("diff acc","diff loss",'Location','southeast')
xlabel('Epoch')
ylabel('Acc')
%}
end
% -------------------------------------------------------------
% -------------------------------------------------------------
% -------------------------------------------------------------
%path ='/home/agrot12/Dropbox/AIReport/figures/';
%filenameFigure = strcat(path,saveNames{re},'.png');

%saveas(gcf,filenameFigure)

figure('Name','Acc')
for ch = 1:changed
    plot(epoch, acc_mean(:,ch));
    hold on 
end
title('Acc');
xlabel('Epoch')
ylabel('Acc')
legend(titels,'Location','southeast')


figure('Name','Acc val')
for ch = 1:changed
    plot(epoch, acc_val_mean(:,ch));
    hold on 
end
title('Acc val');
xlabel('Epoch')
ylabel('Acc')
legend(titels,'Location','southeast')


figure('Name','Loss')
for ch = 1:changed
    plot(epoch, loss_mean(:,ch));
    hold on 
end
title('Loss');
xlabel('Epoch')
ylabel('Loss')
legend(titels,'Location','southeast')


figure('Name','Loss val')
for ch = 1:changed
    plot(epoch, loss_val_mean(:,ch));
    hold on 
end
title('Loss val');
xlabel('Epoch')
ylabel('Loss')
legend(titels,'Location','southeast')

    