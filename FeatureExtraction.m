% clc;
clear all;
 prefix1 = "50_wisig_data/";
 devNum = 50;  %设备数量
 Frame = 240;
 feature = [1:6];


CFO = zeros(Frame,devNum);
PhaseOffset = zeros(38,Frame,devNum);
cpe_data = zeros(38,Frame,devNum);
M = 64;
x = 0:M-1;
symbols = qammod(x,M,'UnitAveragePower', true); 

Feature = zeros(130,6,devNum);

tt = 0;
 for dev_num = 40:40 %[3 8]
     tt = tt + 1;
     data_count=1;
     filename = sprintf('%s%d.mat', prefix1, dev_num);
    if ~exist(filename, 'file')
        disp(['当前文件不存在：', filename]);
        return ;
    end
    data = load(filename);

    new_folder =sprintf('设备%d',dev_num);

    % Create the full path of the new folder


    % Check if the folder already exists
    % if ~exist(new_folder, 'dir')
    %    % If not, create it
    %    mkdir(new_folder);
    % end
    
    a = length(data.signal_array(:,1));
    for k=1:Frame
        try
        rx=data.signal_array(k,:).';
        rx_new = resample(rx,4,5);
        % [CFO(k,tt),PhaseOffset(:,k,tt),cpe_data(:,k,tt),eqSym_data, errorPhase_mean(k,tt),errorMagnitude_mean(k,tt),rpsd] = data_process3(rx_new);
        [CFO(k,tt),errorPhase_mean(k,tt),errorMagnitude_mean(k,tt),errorPhaseVar(k,tt),errorMagnitudeVar(k,tt)] = data_process3(rx_new);
        Feature(data_count,:,tt) = [CFO(k,tt)/20e6,errorPhase_mean(k,tt),errorMagnitude_mean(k,tt),errorPhaseVar(k,tt),errorMagnitudeVar(k,tt),tt];
        disp('成功存储');
        data_count=data_count+1;
        if(data_count>130)
            break
        end
        % if(k>1)
        %     break;
        % end
        catch ME
            disp('未成功存储');
            continue
        end
       % figure('visible','off');
       % plot(real(eqSym_data),imag(eqSym_data),'.');
       % hold on;plot(real(symbols),imag(symbols),'k.',MarkerSize=20);
       % path = ['设备%d/' sprintf('%d',dev_num),sprintf('%d',k) '星座图.png'];
       % path = sprintf('设备%d/星座图%d.png',dev_num,k);
       % saveas(gcf,path); %print(gcf,'-dpng',path); 
       % figure('visible','off');
       % plot(real(data_syms_out),imag(data_syms_out),'.');
       % path = sprintf('设备%d/自己画星座图%d.png',dev_num,k);
       % saveas(gcf,path); %print(gcf,'-dpng',path);
       
    end
    feature = vertcat(feature, Feature(:,:,tt));
    a = sprintf("处理完设备%d", tt);
    disp(a);
    
 end
 % save('50_feature_5z.mat','Feature');
 % csvwrite('50_output32_5z.csv', feature);

 % save('feature_6.mat','Feature1');
 % csvwrite('output32_6.csv', feature1);
 % save('feature_9.mat','Feature2');
 % csvwrite('output32_9.csv', feature2);
 

 %Frame_fine([3,15,29,34,],:) = [];
 % Frame_fine = Frame_fine(1:130,:);