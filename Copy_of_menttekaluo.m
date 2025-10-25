clear;
% close all;
tic;          % 启动计时器
%
%% 设置参数
prefix1 = "feature";
numFiles = 15; % 合法设备总数，编号1：K
allFiles = 27; % 设备总数量
Ka = allFiles;
K = numFiles;
tmp = [130, 4]; %num的大小
featureNum = tmp(2)-1;
percent = 0.8; % 训练集占比1
KNN_K = 3;
MSC = 0.90;  %自体覆盖率 (Self-Coverage Rate, 期望自体被检测器排除的比例)
c0 = 0.99; %期望非自体覆盖率 (Expected Non-Self Coverage Rate)
MaxNumDet = 3000; % 最大检测器数量
rs = 0.008;%可变半径阈值 
Gen = 40;%进化代数 (每次抗体生成迭代的进化次数，此Gen将变为每次"尝试”生成检测器的内部代数)
N0 =25;%初始检测器数量

%% --- START MODIFICATION ---
% 定义蒙特卡洛抽样次数，用于检测自体覆盖率
num_mc_samples_self_coverage = 1000; % 用于评估当前检测器集合对自体数据的覆盖率
% 定义蒙特卡洛抽样次数，用于 clac_coverage 函数，评估非自体空间覆盖率
num_mc_points_for_clac_coverage = 5000; % 确保 clac_coverage 函数使用这个值
%% --- END MODIFICATION ---

%% %% 可只运行1次
% 构建训练集和测试集
tran_set_size = ceil(percent*tmp(1));   %  ----130*0.8=104   tran_set_size = 104
test_set_size = ceil((1-percent)*tmp(1)); %----26   test_set_size = 26
tran_set = zeros(tran_set_size, tmp(2), Ka);
test_set = zeros(test_set_size, tmp(2)+1, Ka); %测试集合多1列， ---用于存储预测标签
load('compare_stable_data_16_008_98ill.mat');  % choose_f32_n5_z
% load('paperdata.mat');
% feature32_2dim_mag_norm   feature24_2dim_mag_norm_chan2
data=reorderedData;
data_normalized = data;
% round_i = 1;
% % 
% paper_data = zeros(50,7,11);

% while (numFiles <= 15)
% while (round_i < 51 )
%设置训练集、测试集
for i=1:Ka
    randIdx = randperm(tmp(1)); % 生成随机索引数组
    %再次打乱分配
    data_normalized(:,:,i) = data_normalized(randIdx, :,i);
    tran_set(1:tran_set_size, 1:tmp(2), i) = data_normalized(1:tran_set_size, :,i);
    test_set(1:tmp(1)-tran_set_size, 1:tmp(2), i) = data_normalized(tran_set_size+1:end, :,i);
end
% 删除训练集中非法设备特征
tran_set = tran_set(:, :, 1:K);
%打乱测试集
test_set_new = zeros(test_set_size*Ka,featureNum+2);
random_num = randperm(test_set_size*Ka);
for k=1:Ka
    for kk = 1:test_set_size
        if k>K
            test_set_new(random_num((k-1)*test_set_size+kk), 1:featureNum) = test_set(kk,1:featureNum,k);
            test_set_new(random_num((k-1)*test_set_size+kk), featureNum+1) = K+1; %恶意类别编号是K+1
        else
            test_set_new(random_num((k-1)*test_set_size+kk),:) = test_set(kk,:,k);
        end
        
    end
end
%% 设置可变自体半径
% 计算各合法设备之间自体的距离
distance = zeros(K*tran_set_size,(K-1)*tran_set_size+1);
for k=1:K
    for n = 1:tran_set_size
        mm = 0;
        for kk = 1:K
            if(kk~=k)
                for nn = 1:tran_set_size
                    mm = mm + 1;
                    distance((k-1)*tran_set_size+n,mm)= norm(data_normalized(n,1:featureNum,k)-data_normalized(nn,1:featureNum,kk));
                end
                    distance((k-1)*tran_set_size+n,mm+1) = k;
            end
        end
    end
end
% 计算同一设备自体类内距离
distance_internal = zeros(tran_set_size,tran_set_size,K);
for k = 1:K
    for n = 1:tran_set_size
        for nn = 1:tran_set_size
            distance_internal(n,nn,k) = norm(data_normalized(n,1:featureNum,k)-data_normalized(nn,1:featureNum,k));
            if(n==nn)
                distance_internal(n,nn,k) = inf;
            end
        end
    end
end
%设置可变自体半径 
r_self = zeros(K,tran_set_size);
% 计算自体半径
for k = 1:K
    for n = 1:tran_set_size
        %自体半径设置为：？
        r_self(k,n) = min(min(min(distance((k-1)*tran_set_size+n,1:(K-1)*tran_set_size))/2,2*min(distance_internal(n,:,k))),rs); 
        % r_self(k,n) = min(min(distance((k-1)*tran_set_size+n,1:(K-1)*tran_set_size))/2,2*min(distance_internal(n,:,k)));
        %%%改
        % r_self(k,n) = rs;
    end
end

% r_self(:,:) = 0.010;

%% 检测器生成
% 存储成熟检测器
detector_tol = zeros(MaxNumDet,featureNum+1,K);
% 统计每个设备对应检测器集合的检测器个数
detectorTolNum = zeros(1,K);



%% 1 对每个类别生成检测器
for k =1:K
    %% --- START MODIFICATION ---
    % 为每个类别引入一个循环，直到满足覆盖率要求或达到最大检测器数量
    detector_generation_done_for_k = false;
    
    % 初始化当前设备的检测器集合和计数
    % 这里我们使用一个动态增长的数组，或者一个临时数组，然后一次性复制到 detector_tol
    % 为了避免重复分配大数组，我们使用一个临时列表，最后再赋值
    % 或者更直接地，使用 MaxNumDet 来限制 current_k_detectors 的大小
    current_k_detectors_temp = zeros(MaxNumDet, featureNum + 1); % 临时存储当前设备的检测器
    current_k_detector_count = 0;

    % 在外层循环中持续尝试生成检测器
    iter_attempt = 0; % 记录尝试生成检测器的次数
    % 修改 while 循环条件，确保不会无限尝试，并考虑 MaxNumDet 限制
    while ~detector_generation_done_for_k  && iter_attempt < 20% 增加一个尝试次数的上限
        iter_attempt = iter_attempt + 1;

        % 每次尝试时，重新初始化抗体，或者从上次的精英抗体开始，这里我们简化为重新开始
        M = N0+8; % 初始检测器数量
        new_gen_detector = rand(M,featureNum); % 初始化M个检测器
        if (iter_attempt==1)
            new_gen_detector(26,:)=[0,0,0];
            new_gen_detector(27,:)=[0,1,0];
            new_gen_detector(28,:)=[0,0,1];
            new_gen_detector(29,:)=[0,1,1];
            new_gen_detector(30,:)=[1,0,0];
            new_gen_detector(31,:)=[1,0,1];
            new_gen_detector(32,:)=[1,1,0];
            new_gen_detector(33,:)=[1,1,1];
        end
        gen = 0; % 迭代次数
        
        %% 1.1 基于自体分布生成抗体，迭代次数 (此内部循环保持不变，控制抗体进化)
        while(gen<Gen)            
            %% 1.1.1 克隆 变异
            distance_det_and_self = [];
            for n=1:M
                distance_det_and_self(n) = min(sqrt(sum((new_gen_detector(n,:) - tran_set(:,1:featureNum,k)).^2,2)));
            end
            cloneNum = [];
            cloneNum = zeros(1,M);
            numDetMuate = 0;
            muate_coff = [];
            muate_coff = zeros(1,M);
            detector_muate = zeros(0,featureNum);
           
            min_distance_mean = mean(distance_det_and_self);
            a = pdist2(new_gen_detector,tran_set(:,1:featureNum,k));
            b = mean(a(:));
            affi = 0.4*min_distance_mean; 
            for n = 1:M
                if(distance_det_and_self(n)<affi)
                    cloneNum(n) = 10;
                    muate_coff(n) = distance_det_and_self(n);
                    numDetMuate = numDetMuate + 1;
                    detector_muate(numDetMuate,:) = new_gen_detector(n,:);
                    for nn = 1:cloneNum(n)
                        numDetMuate = numDetMuate + 1;
                        ori = 2*randi([0,1],1,featureNum)-1;
                        detector_muate(numDetMuate,:) = min(max(new_gen_detector(n,:) + muate_coff(n)*rand(1,featureNum).*ori,0),1);
                    end
                elseif(distance_det_and_self(n)>=affi && distance_det_and_self(n)<2*affi)    
                    cloneNum(n) = 5;
                    muate_coff(n) = 2*affi;
                    numDetMuate = numDetMuate + 1;
                    detector_muate(numDetMuate,:) = new_gen_detector(n,:);
                    for nn = 1:cloneNum(n)
                        numDetMuate = numDetMuate + 1;
                        ori = 2*randi([0,1],1,featureNum)-1;
                        detector_muate(numDetMuate,:) = min(max(new_gen_detector(n,:) + muate_coff(n)*rand(1,featureNum).*ori,0),1);
                    end
                else
                    cloneNum(n) = 2;
                    muate_coff(n) = 2*ceil(distance_det_and_self(n)/affi)*affi;
                    numDetMuate = numDetMuate + 1;
                    detector_muate(numDetMuate,:) = new_gen_detector(n,:);
                    for nn = 1:cloneNum(n)
                        numDetMuate = numDetMuate + 1;
                        ori = 2*randi([0,1],1,featureNum)-1;
                        detector_muate(numDetMuate,:) = min(max(new_gen_detector(n,:) + muate_coff(n)*rand(1,featureNum).*ori,0),1);
                    end
                end 
            end
           
            %% 1.1.2 抑制
            distance_det_and_self = [];
            detector_muate_tmp = detector_muate;
            detector_muate_distance = [];
            for n = 1:numDetMuate
                    detector_muate_distance(n,:) = sqrt(sum((detector_muate(n,1:featureNum)-detector_muate(:,1:featureNum)).^2,2));
                    detector_muate_distance(n,n) = 0;
            end
            sp = mean(detector_muate_distance(:))*0.3;
            
            index_tmp = 1;
            while(index_tmp)
                detector_muate_distance = [];
                delete_index = zeros(1,numDetMuate);
                detector_muate = detector_muate_tmp;
                tt_flag = 0;
                
                for n = 1:numDetMuate
                    detector_muate_distance(n,:) = sqrt(sum((detector_muate(n,1:featureNum)-detector_muate(:,1:featureNum)).^2,2));
                    detector_muate_distance(n,n) = inf; 
               
                    for nn = n+1:numDetMuate
                        if(detector_muate_distance(n,nn)<sp)
                            d1 = min(sqrt(sum((detector_muate(n,1:featureNum)-tran_set(:,1:featureNum,k)).^2,2)));
                            d2 = min(sqrt(sum((detector_muate(nn,1:featureNum)-tran_set(:,1:featureNum,k)).^2,2)));
                            if(d1<d2)
                                delete_index(nn) = 1;
                                tt_flag = 1;
                            else
                                delete_index(n) = 1; 
                                tt_flag = 1;
                            end
                            tt=find(delete_index==0);
                            detector_muate_tmp = [];
                            detector_muate_tmp=detector_muate(tt,:);
                            numDetMuate = numDetMuate -1;
                            break;
                        end     
                    end
                    if(tt_flag==1)
                        break;
                    end
                end
                if(tt_flag ==0)
                    index_tmp = 0;
                end
            end
            %% 1.1.3 对抑制后的抗体进行筛选，留下亲和力最高的个体
            new_gen_detector_tmp = detector_muate_tmp(1:numDetMuate ,:);
            distance_muate_det_and_self = [];
            for n=1:numDetMuate
                distance_muate_det_and_self(n) = min(sqrt(sum((new_gen_detector_tmp(n,:) - tran_set(:,1:featureNum,k)).^2,2)));
            end
            [tttt, det_pos_high] = sort(distance_muate_det_and_self);
          
            new_gen_detector = [];
            M = N0;
            M = min(M,numDetMuate);
            new_gen_detector(1:M,1:featureNum) =  new_gen_detector_tmp(det_pos_high(1:M),:);
            gen = gen + 1;
        end
        
        %% 1.2 基于自体密度进化 (此部分也应在每次尝试中执行)
        numDetMuate = M;
        detector_muate = new_gen_detector;
        distance_det_and_self = pdist2(detector_muate,tran_set(:,1:featureNum,k));
        dis_avg = mean(distance_det_and_self(:));
        density = zeros(1,M);
        for i = 1:M
            for j = 1:tran_set_size
                if distance_det_and_self(i,j) < dis_avg
                    density(i) = density(i)+1;
                end
            end
        end
        density_avg = mean(density);
        cloneNum = [];
        tmp_num = numDetMuate;
        detector_muate_new = detector_muate;
        
        for n = 1:tmp_num
            if(density(n)>=density_avg)
                cloneNum(n) = 10;
                muate_coff(n) = min(distance_det_and_self(n,:));
                for nn = 1:cloneNum(n)
                    numDetMuate = numDetMuate + 1;
                    ori = 2*randi([0,1],1,featureNum)-1;
                    detector_muate_new(numDetMuate,:) = min(max(detector_muate(n,:) + muate_coff(n)*rand(1,featureNum).*ori,0),1);
               end
            elseif(density(n)>density_avg*0.5 && density(n)<density_avg)
                cloneNum(n) = 5;
                muate_coff(n) = 0.5*max(distance_det_and_self(n,:));
                for nn = 1:cloneNum(n)
                    numDetMuate = numDetMuate + 1;
                    ori = 2*randi([0,1],1,featureNum)-1;
                    detector_muate_new(numDetMuate,:) = min(max(detector_muate(n,:) + muate_coff(n)*rand(1,featureNum).*ori,0),1);
               end
            else
                cloneNum(n) = 3;
                muate_coff(n) = 1*max(distance_det_and_self(n,:));
                for nn = 1:cloneNum(n)
                    numDetMuate = numDetMuate + 1;
                    ori = 2*randi([0,1],1,featureNum)-1;
                    detector_muate_new(numDetMuate,:) = min(max(detector_muate(n,:) + muate_coff(n)*rand(1,featureNum).*ori,0),1);
               end
            end
        end
        
        %% 1.2.2 抑制
        den_radius =  mean(pdist(tran_set(:,1:featureNum,k)));
        distance_det_and_det = pdist2(detector_muate_new,detector_muate_new);
        distance_det_and_self = pdist2(detector_muate_new,tran_set(:,1:featureNum,k));
        dis_avg = mean(distance_det_and_self(:));
        density = zeros(1, numDetMuate);
        for i = 1:numDetMuate
            for j = 1:tran_set_size
                if distance_det_and_self(i,j)<dis_avg
                    density(i) = density(i)+1;
                end
            end
        end
        density_avg = mean(density);
        
        index_tmp = 1;
        detector_muate_tmp = detector_muate_new;
        while(index_tmp)
            detector_muate_distance  = pdist2(detector_muate_new,detector_muate_new);
            detector_muate_distance = detector_muate_distance+10000*eye(numDetMuate);
            sp = [];
            delete_index = zeros(1,numDetMuate);
            detector_muate_new = detector_muate_tmp;
            
            distance_det_and_self = pdist2(detector_muate_new,tran_set(:,1:featureNum,k));
            tt_flag = 0;
            
            for n=1:numDetMuate
                 if(density(n)<0.2*density_avg)
                    sp(n) = 0.1*den_radius; 
                elseif(density(n)<density_avg&&density(n)>=0.2*density_avg)
                    sp(n) = 0.8*den_radius;
                else
                    sp(n) = 0.9*den_radius;
                end
        
                for nn = n+1:numDetMuate
                    if(detector_muate_distance(n,nn)<sp(n))
                        d1 = min(distance_det_and_self(n,:));
                        d2 = min(distance_det_and_self(nn,:));
                        if(d1<d2)
                            delete_index(nn) = 1;
                            tt_flag = 1;
                        else
                            delete_index(n) = 1; 
                            tt_flag = 1;
                        end
                        tt=find(delete_index==0);
                        detector_muate_tmp = [];
                        detector_muate_tmp=detector_muate_new(tt,:);
                        numDetMuate = numDetMuate -1;
                        break;
                    end     
                end
                if(tt_flag==1)
                   break;
                end
            end
            if(tt_flag ==0)
               index_tmp = 0;
            end
            detector_muate_new = detector_muate_tmp;
        end
        
        %% 根据抗体生成检测器
        % ts = 0; % 此处 ts 变量不再用于控制循环，而是记录当前未被覆盖的自体点数量
        if (iter_attempt > 8)
            
            detector_muate_new(numDetMuate+1,:)=[0,0,0];
            detector_muate_new(numDetMuate+2,:)=[0,0,1];
            detector_muate_new(numDetMuate+3,:)=[0,1,0];
            detector_muate_new(numDetMuate+4,:)=[0,1,1];
            detector_muate_new(numDetMuate+5,:)=[1,0,0];
            detector_muate_new(numDetMuate+6,:)=[1,0,1];
            detector_muate_new(numDetMuate+7,:)=[1,1,0];
            detector_muate_new(numDetMuate+8,:)=[1,1,1];
            numDetMuate = numDetMuate+8;
        end
        
        num_new_matured_this_attempt = 0; % 记录本次尝试新生成的成熟检测器数量
        for nn=1:numDetMuate % 遍历进化和抑制后的抗体
            detector_tmp = detector_muate_new(nn,:);
            
            flag= 1; % 1表示检测器可以使用
            
            if flag == 1
                min_d=inf; % 重置最短距离
                min_m = 0;
                for m  = 1:1:tran_set_size
                    dist = norm(tran_set(m,1:featureNum,k)-detector_tmp);
                    if min_d > dist
                        min_d = dist;
                        min_m = m;
                    end
                end
        
                % 如果最小距离大于自体半径，则认为是一个成熟检测器
                if min_d > r_self(k,min_m)             
                    % 在这里检查是否超过 MaxNumDet，如果超过则不添加
                    if current_k_detector_count < MaxNumDet
                        current_k_detector_count = current_k_detector_count + 1; % 成熟的检测器数目+1            
                        detector_radius_tmp = min_d - r_self(k,min_m); % 计算检测器半径
                        current_k_detectors_temp(current_k_detector_count, 1:featureNum) = detector_tmp; % 保存检测器信息
                        current_k_detectors_temp(current_k_detector_count, end) = detector_radius_tmp; % 保存检测器半径
                        num_new_matured_this_attempt = num_new_matured_this_attempt + 1;
                    else
                        % 达到最大检测器数量，停止当前检测器生成尝试
                        % detector_generation_done_for_k = true; % 这会导致整个设备生成停止
                        break; % 跳出当前 nn 循环，不再处理当前批次剩余的抗体
                    end
                end % End if min_d > r_self
            end % End if flag == 1
        end % End for nn=1:numDetMuate
        
        % 每次尝试结束后，重新评估当前设备的总检测器集合的覆盖率
        % 这些检测器都应该存储在 current_k_detectors_temp 中
        
        % 检查停止条件
        % 1. 非自体空间覆盖率
        fc = clac_coverage(current_k_detectors_temp(1:current_k_detector_count,1:featureNum),...
                           current_k_detectors_temp(1:current_k_detector_count,end),...
                           current_k_detector_count,featureNum,c0, num_mc_points_for_clac_coverage);
        expected_coverage_threshold = 1 / (1 - c0); % 期望覆盖率阈值
        disp("expected_coverage_threshold"+fc);
        
        % 2. 自体覆盖率（蒙特卡洛检测）
        covered_self_points_for_msc = 0; % 记录自体被检测器“排斥”的点的数量
        if current_k_detector_count > 0 % 只有当有检测器时才计算自体覆盖率
            for mc_idx = 1:num_mc_samples_self_coverage
                rand_self_idx = randi(tran_set_size);
                self_point = tran_set(rand_self_idx, 1:featureNum, k);
                
                is_covered_by_any_detector = false;
                for d_idx = 1:current_k_detector_count
                    detector_center = current_k_detectors_temp(d_idx, 1:featureNum);
                    detector_radius = current_k_detectors_temp(d_idx, end); 
                    
                    if norm(self_point - detector_center) < detector_radius
                        is_covered_by_any_detector = true;
                        break;
                    end
                end
                % 自体覆盖率是衡量自体有多少点 **未** 被检测器覆盖（即被排斥）
                if ~is_covered_by_any_detector 
                    covered_self_points_for_msc = covered_self_points_for_msc + 1;
                end
            end
        end
        current_self_rejection_rate = covered_self_points_for_msc / num_mc_samples_self_coverage;

        
        % 最终判断是否满足停止条件
        if (fc >= c0 && current_self_rejection_rate >= MSC) || ...
           current_k_detector_count >= MaxNumDet 
            detector_generation_done_for_k = true;
        end
        
        % 添加一个安全机制，防止无限循环
        if iter_attempt >= MaxNumDet * 10 % 比如，尝试次数超过最大检测器数量的10倍，就强制停止
            fprintf('Warning: Detector generation for device %d stopped due to excessive attempts (%d). MaxNumDet=%d\n', k, iter_attempt, MaxNumDet);
            detector_generation_done_for_k = true;
        end

    end % End while ~detector_generation_done_for_k
    
    % 将最终生成的检测器从临时数组复制到 detector_tol
    detector_tol(1:current_k_detector_count, :, k) = current_k_detectors_temp(1:current_k_detector_count, :);
    detectorTolNum(k) = current_k_detector_count; % 更新总数量
    
    %% --- END MODIFICATION ---

    a = sprintf("设备%d检测器已生成完毕",k);
    disp(a);
    % % 画图
    % figure;plot(tran_set(:,1,k),tran_set(:,2,k),'.','Color','r');hold on;
    % plot(detector_tol(:,1,k),detector_tol(:,2,k),'k.');
    % axis([0 1 0 1]);
end


%% 测试集验证
    %对测试集
    for kk = 1:test_set_size*Ka %kk：每个测试集个体
        %对每个合法设备的检测器集合
        random_K = randperm(K);
        %设置检测变量
        detectnum = 0; %标志量，未被检测器捕获的类别总数
        detects = [];
        for kkk = random_K %kkk：每个检测器类别
            %对每个检测器
            dev_flag = 1; % 标志量，1表示未被检测器捕获，0表示被检测器捕获
            for kkkk = 1:1:detectorTolNum(kkk)
                %先到先得到
                if(norm(detector_tol(kkkk,1:featureNum,kkk)-test_set_new(kk,1:featureNum))<detector_tol(kkkk,end,kkk))
                    dev_flag = 0;
                    break;
                end
            end
            if(dev_flag==1) %未被捕获
                detects = vertcat(detects, [kkk]);
                detectnum = detectnum + 1;
                % test_set_new(kk,end) = kkk;
                % % a = sprintf("当前设备被识别为：%d,当前设备实际为：%d",kkk-1,k-1);
                % % disp(a);
                % break; 
            end
        end
        
        if detectnum == 0 % 被所有检测器捕获，则是非法类别
            test_set_new(kk,end) = K+1;
        end
        if detectnum == 1 % 若只未被1类检测器捕获，则判别为该类别
            test_set_new(kk,end) = detects(1);
        end
        if detectnum > 1 % 通过距离计算类别
            
            mean_val = zeros(detectnum,2);
            for iii = 1:detectnum
                distance11 = sqrt(sum((test_set_new(kk,1:featureNum)-detector_tol(1:detectorTolNum(detects(iii)),1:featureNum,detects(iii))).^2,2));
                [dis,position] = sort(distance11);
                mean_val(iii,1) = mean(dis(1:KNN_K));
                mean_val(iii,2) = detects(iii);
            end
            [row, ~] = find(mean_val(:, 1) == min(mean_val(:, 1)));
            test_set_new(kk,end) = mean_val(row,2);    
                 
        end
        
    end
%计算混淆矩阵
confusion_matrix = zeros(K+1,K+1);
    for kk = 1:test_set_size*Ka
        if test_set_new(kk,featureNum+1)==0
            continue;
        end
        if(test_set_new(kk,featureNum+1)==test_set_new(kk,end)) %如果真实类=预测类，合法判断正确
           confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,featureNum+1)) = confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,featureNum+1)) + 1;
        else
            confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,end)) = confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,end)) + 1;
        end
    end
accuray = sum(diag(confusion_matrix ))/(test_set_size*Ka);
b = sprintf("*********整体识别准确率为：%f***********",accuray);
disp(b);
letimate_accuray = (sum(diag(confusion_matrix))-confusion_matrix(K+1,K+1))/(test_set_size*K);
b = sprintf("*********合法设备识别准确率为：%f***********",letimate_accuray);
disp(b);
ill_accuray = confusion_matrix(K+1,K+1)/(test_set_size*(Ka-K));
b = sprintf("*********恶意设备识别准确率为：%f***********",ill_accuray);
disp(b);
figure();
h = heatmap(confusion_matrix);
title("基于距离计算的合法设备分类混淆矩阵");
h.Colormap = jet;
elapsed_time = toc;  
fprintf('模型运行时间: %.4f 秒\n', elapsed_time); 
%% --- （以下部分直接追加到原脚本末尾） ---
% 计算各类别的精确率、召回率和F1分数
numClasses = K + 1;  % 类别总数（包括恶意类别）
precision = zeros(1, numClasses);
recall    = zeros(1, numClasses);
f1        = zeros(1, numClasses);
for i = 1:numClasses
    tp = confusion_matrix(i, i);                         % 真正例
    fp = sum(confusion_matrix(:, i)) - tp;                % 假正例
    fn = sum(confusion_matrix(i, :)) - tp;                % 假反例
    if tp + fp == 0
        precision(i) = NaN;  % 避免除零
    else
        precision(i) = tp / (tp + fp);
    end
    if tp + fn == 0
        recall(i) = NaN;
    else
        recall(i) = tp / (tp + fn);
    end
    if precision(i) + recall(i) == 0
        f1(i) = NaN;
    else
        f1(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));
    end
end
% 输出每个类别的指标
fprintf('\n--- 各类别性能指标 ---\n');
for i = 1:numClasses
    if i <= K
        label = sprintf('合法设备 %d', i);
    else
        label = '恶意设备';
    end
    fprintf('%-10s：Precision = %.4f, Recall = %.4f, F1 = %.4f\n', ...
            label, precision(i), recall(i), f1(i));
end
% 计算宏平均（Macro-average）指标
valid = ~isnan(precision) & ~isnan(recall);
macroP = mean(precision(valid));
macroR = mean(recall(valid));
macroF1 = mean(f1(valid));
fprintf('\n宏平均指标：Precision = %.4f, Recall = %.4f, F1 = %.4f\n', ...
        macroP, macroR, macroF1);
%% --- （在计算完 precision/recall/f1 之后，追加以下部分） ---
% 计算各类别的 FPR（误报率）和 FNR（漏报率）
fpr = zeros(1, numClasses);
fnr = zeros(1, numClasses);
totalSamples = sum(confusion_matrix(:));
for i = 1:numClasses
    tp = confusion_matrix(i, i);
    fp = sum(confusion_matrix(:, i)) - tp;
    fn = sum(confusion_matrix(i, :)) - tp;
    % 计算 True Negatives
    tn = totalSamples - tp - fp - fn;
    
    % FPR = FP / (FP + TN)
    if fp + tn == 0
        fpr(i) = NaN;
    else
        fpr(i) = fp / (fp + tn);
    end
    
    % FNR = FN / (TP + FN)
    if tp + fn == 0
        fnr(i) = NaN;
    else
        fnr(i) = fn / (tp + fn);
    end
end
% 输出每个类别的 FPR 和 FNR
fprintf('\n--- 各类别误报率 (FPR) 和漏报率 (FNR) ---\n');
for i = 1:numClasses
    if i <= K
        label = sprintf('合法设备 %d', i);
    else
        label = '恶意设备';
    end
    fprintf('%-10s：FPR = %.4f, FNR = %.4f\n', ...
            label, fpr(i), fnr(i));
end
% 计算宏平均（Macro-average）FPR 和 FNR
validFPR = ~isnan(fpr);
validFNR = ~isnan(fnr);
macroFPR = mean(fpr(validFPR));
macroFNR = mean(fnr(validFNR));
fprintf('\n宏平均误报率 (Macro FPR) = %.4f, 漏报率 (Macro FNR) = %.4f\n', ...
        macroFPR, macroFNR);
letimate_FPR=fpr(1:end-1);
letimate_FNR=fnr(1:end-1);
letimate_validFPR=~isnan(letimate_FPR);
letimate_validFNR=~isnan(letimate_FNR);
letimate_macroFPR = mean(letimate_FPR(letimate_validFPR));
letimate_macroFNR = mean(letimate_FNR(letimate_validFPR));
fprintf('\n合法宏平均误报率 (Macro FPR) = %.4f, 合法漏报率 (Macro FNR) = %.4f\n', ...
        letimate_macroFPR, letimate_macroFNR);

% 
% paper_data(round_i,:,KNN_K) = [accuray,letimate_accuray,ill_accuray,letimate_macroFPR,letimate_macroFNR,macroFPR,macroFNR];
% 
% round_i =round_i +1;
% 
% 
% fprintf("\n现在是第%d轮，numFiles=%d\n",round_i,numFiles);
% end
% round_i = 1;
% numFiles = numFiles +1;
% end
% save('self_numFiles_data.mat','paper_data');



% 修改 clac_coverage 函数，使其接受蒙特卡洛抽样次数作为参数
function [cov]=clac_coverage(Detector,Dr,num_Detector,featureNum,pexp, num_mc_points)
    cov=0;
    if num_Detector == 0
        cov = 0; % 如果没有检测器，覆盖率为0
        return;
    end
    
    for mc_idx = 1:num_mc_points % 每次抽样一个点
       temp_c = rand(1,featureNum); % 确保随机点在0-1范围内，与您的归一化数据一致
       
       is_covered = false;
       for i=1:num_Detector
            dis = sqrt(sum((Detector(i,:)-temp_c).^2));
            if dis < Dr(i) % 如果点在某个检测器内
               is_covered = true;
               break; % 只要被一个检测器覆盖，就认为该点被覆盖
            end
       end
       
       if is_covered
           cov = cov + 1;
       end
    end
    
    cov = cov / num_mc_points; % 计算实际覆盖率
end

