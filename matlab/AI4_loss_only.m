clc, clear, close all;%, close all;
nr = 1;
seed = 1;
titels = {};

type = "AmountOfImages/";
%type = "LearningRate/";
%type = "Solver/";

if type == "LearningRate/" 
    titels  = {'0.2','0.4','0.6','0.8'};
elseif type == "Solver/"
    titels  = {'SGD','Adadelta','Adam'};
elseif type == "AmountOfImages/"
    titels  = {'train1700val200','train3400val400','train5100val600','train6800val800'};
end



folder = '../results/'+type;


filenameInit = strcat(folder,titels{nr},'_seed',int2str(seed),'.txt');
dataInit = importdata(filenameInit);



epochs = size(dataInit);
epochs = epochs(1);
seeds = 1; %4
changed = size(titels);
changed = changed(2);


lossA = zeros([epochs seeds]);
loss_valA = zeros([epochs seeds]);

loss_mean = zeros([epochs changed]);
loss_val_mean = zeros([epochs changed]);

for nr=1:changed
    for seed=1:seeds

        filename = strcat(folder,titels{nr},'_seed',int2str(seed),'.txt');

        data = importdata(filename);
        epoch = data(:,1);
        loss = data(:,4);
        loss_val = data(:,5);

        lossA(:,seed)=loss;
        loss_valA(:,seed)=loss_val;

    end
    
    lossAll{nr} = lossA;
    loss_valAll{nr} = loss_valA;
%end

% -------------------------------------------------------------
% Val----------------------------------------------------------
% -------------------------------------------------------------

mean_lossA = mean(lossA')';
std_lossA = std(lossA')';
loss_mean(:,nr) = lossA;%mean_lossA;

mean_loss_valA = mean(loss_valA')';
std_loss_valA = std(loss_valA')';
loss_val_mean(:,nr) = loss_valA;%mean_loss_valA;

%{

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
title('std');
legend("std loss","std loss val",'Location','northwest')
ylabel('std')
xlabel('Epoch')
% -------------------------------------------------------------
% Difference ----------------------------------------------------------
% -------------------------------------------------------------

diff_lossA = mean_loss_valA-mean_lossA;



figure('Name',titels{nr});
plot(epoch,diff_lossA)
hold on;
title(titels{nr});
legend("diff loss",'Location','southeast')
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

    