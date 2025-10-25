clear;

tic;    

prefix1 = "feature";
numFiles = 15; 
allFiles = 27; 
Ka = allFiles;
K = numFiles;
tmp = [130, 6]; 
featureNum = tmp(2)-1;
percent = 0.8; 
KNN_K = 3;
MSC = 0.90; 
c0 = 0.99; 
MaxNumDet = 3000; 
Gen = 40;
N0 =25;


num_mc_samples_self_coverage = 1000; 

num_mc_points_for_clac_coverage = 5000; 

tran_set_size = ceil(percent*tmp(1));   
test_set_size = ceil((1-percent)*tmp(1));
tran_set = zeros(tran_set_size, tmp(2), Ka);
test_set = zeros(test_set_size, tmp(2)+1, Ka); 
load('Dataset.mat');  

data=Feature;
data_normalized = data;

for i=1:Ka
    randIdx = randperm(tmp(1)); 
    data_normalized(:,:,i) = data_normalized(randIdx, :,i);
    tran_set(1:tran_set_size, 1:tmp(2), i) = data_normalized(1:tran_set_size, :,i);
    test_set(1:tmp(1)-tran_set_size, 1:tmp(2), i) = data_normalized(tran_set_size+1:end, :,i);
end
tran_set = tran_set(:, :, 1:K);
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

r_self = zeros(K,tran_set_size);

for k = 1:K
    for n = 1:tran_set_size
        r_self(k,n) = min(min(min(distance((k-1)*tran_set_size+n,1:(K-1)*tran_set_size))/2,2*min(distance_internal(n,:,k))),rs); 
    end
end
detector_tol = zeros(MaxNumDet,featureNum+1,K);
detectorTolNum = zeros(1,K);
for k =1:K
    detector_generation_done_for_k = false;
    current_k_detectors_temp = zeros(MaxNumDet, featureNum + 1); 
    current_k_detector_count = 0;
    iter_attempt = 0; 
    while ~detector_generation_done_for_k  && iter_attempt < 20
        iter_attempt = iter_attempt + 1;
        M = N0+8; 
        new_gen_detector = rand(M,featureNum); 
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
        gen = 0;
        while(gen<Gen) 
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
      
        
        num_new_matured_this_attempt = 0; 
        for nn=1:numDetMuate 
            detector_tmp = detector_muate_new(nn,:);
            
            flag= 1; 
            
            if flag == 1
                min_d=inf; 
                min_m = 0;
                for m  = 1:1:tran_set_size
                    dist = norm(tran_set(m,1:featureNum,k)-detector_tmp);
                    if min_d > dist
                        min_d = dist;
                        min_m = m;
                    end
                end
        
                
                if min_d > r_self(k,min_m)             
                    
                    if current_k_detector_count < MaxNumDet
                        current_k_detector_count = current_k_detector_count + 1;            
                        detector_radius_tmp = min_d - r_self(k,min_m);
                        current_k_detectors_temp(current_k_detector_count, 1:featureNum) = detector_tmp; 
                        current_k_detectors_temp(current_k_detector_count, end) = detector_radius_tmp; 
                        num_new_matured_this_attempt = num_new_matured_this_attempt + 1;
                    else
                
                        break;
                    end
                end 
            end 
        end 
        
        fc = clac_coverage(current_k_detectors_temp(1:current_k_detector_count,1:featureNum),...
                           current_k_detectors_temp(1:current_k_detector_count,end),...
                           current_k_detector_count,featureNum,c0, num_mc_points_for_clac_coverage);
        expected_coverage_threshold = 1 / (1 - c0); 
        disp("expected_coverage_threshold"+fc);
        
        
        covered_self_points_for_msc = 0; 
        if current_k_detector_count > 0 
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
                if ~is_covered_by_any_detector 
                    covered_self_points_for_msc = covered_self_points_for_msc + 1;
                end
            end
        end
        current_self_rejection_rate = covered_self_points_for_msc / num_mc_samples_self_coverage;
        if (fc >= c0 && current_self_rejection_rate >= MSC) || ...
           current_k_detector_count >= MaxNumDet 
            detector_generation_done_for_k = true;
        end
        

        if iter_attempt >= MaxNumDet * 10 
            fprintf('Warning: Detector generation for device %d stopped due to excessive attempts (%d). MaxNumDet=%d\n', k, iter_attempt, MaxNumDet);
            detector_generation_done_for_k = true;
        end

    end 
    detector_tol(1:current_k_detector_count, :, k) = current_k_detectors_temp(1:current_k_detector_count, :);
    detectorTolNum(k) = current_k_detector_count;

    a = sprintf("Devices %d detector is done",k);
    disp(a);

end



    for kk = 1:test_set_size*Ka 
        
        random_K = randperm(K);
        
        detectnum = 0;
        detects = [];
        for kkk = random_K 
            dev_flag = 1; 
            for kkkk = 1:1:detectorTolNum(kkk)
                if(norm(detector_tol(kkkk,1:featureNum,kkk)-test_set_new(kk,1:featureNum))<detector_tol(kkkk,end,kkk))
                    dev_flag = 0;
                    break;
                end
            end
            if(dev_flag==1) 
                detects = vertcat(detects, [kkk]);
                detectnum = detectnum + 1;
            end
        end
        
        if detectnum == 0 
            test_set_new(kk,end) = K+1;
        end
        if detectnum == 1 
            test_set_new(kk,end) = detects(1);
        end
        if detectnum > 1 
            
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

confusion_matrix = zeros(K+1,K+1);
    for kk = 1:test_set_size*Ka
        if test_set_new(kk,featureNum+1)==0
            continue;
        end
        if(test_set_new(kk,featureNum+1)==test_set_new(kk,end)) 
           confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,featureNum+1)) = confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,featureNum+1)) + 1;
        else
            confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,end)) = confusion_matrix(test_set_new(kk,featureNum+1),test_set_new(kk,end)) + 1;
        end
    end
accuray = sum(diag(confusion_matrix ))/(test_set_size*Ka);
b = sprintf("*********Acc：%f***********",accuray);
disp(b);
letimate_accuray = (sum(diag(confusion_matrix))-confusion_matrix(K+1,K+1))/(test_set_size*K);
b = sprintf("*********ACCl：%f***********",letimate_accuray);
disp(b);
ill_accuray = confusion_matrix(K+1,K+1)/(test_set_size*(Ka-K));
b = sprintf("*********Accu：%f***********",ill_accuray);
disp(b);
figure();
h = heatmap(confusion_matrix);
title("Confusion");
h.Colormap = jet;
elapsed_time = toc;  
fprintf('Running time: %.4f 秒\n', elapsed_time); 

numClasses = K + 1;  
precision = zeros(1, numClasses);
recall    = zeros(1, numClasses);
f1        = zeros(1, numClasses);
for i = 1:numClasses
    tp = confusion_matrix(i, i);                         
    fp = sum(confusion_matrix(:, i)) - tp;              
    fn = sum(confusion_matrix(i, :)) - tp;              
    if tp + fp == 0
        precision(i) = NaN;  
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

for i = 1:numClasses
    if i <= K
        label = sprintf('Legal device%d', i);
    else
        label = 'Unauthorized device';
    end
    fprintf('%-10s：Precision = %.4f, Recall = %.4f, F1 = %.4f\n', ...
            label, precision(i), recall(i), f1(i));
end
valid = ~isnan(precision) & ~isnan(recall);
macroP = mean(precision(valid));
macroR = mean(recall(valid));
macroF1 = mean(f1(valid));
fprintf('\nPrecision = %.4f, Recall = %.4f, F1 = %.4f\n', ...
        macroP, macroR, macroF1);
fpr = zeros(1, numClasses);
fnr = zeros(1, numClasses);
totalSamples = sum(confusion_matrix(:));
for i = 1:numClasses
    tp = confusion_matrix(i, i);
    fp = sum(confusion_matrix(:, i)) - tp;
    fn = sum(confusion_matrix(i, :)) - tp;
    tn = totalSamples - tp - fp - fn;
  
    if fp + tn == 0
        fpr(i) = NaN;
    else
        fpr(i) = fp / (fp + tn);
    end
 
    if tp + fn == 0
        fnr(i) = NaN;
    else
        fnr(i) = fn / (tp + fn);
    end
end

for i = 1:numClasses
    if i <= K
        label = sprintf('Legal device %d', i);
    else
        label = 'Unauthorized device';
    end
    fprintf('%-10s：FPR = %.4f, FNR = %.4f\n', ...
            label, fpr(i), fnr(i));
end

validFPR = ~isnan(fpr);
validFNR = ~isnan(fnr);
macroFPR = mean(fpr(validFPR));
macroFNR = mean(fnr(validFNR));
fprintf('\n (Macro FPR) = %.4f,  (Macro FNR) = %.4f\n', ...
        macroFPR, macroFNR);
letimate_FPR=fpr(1:end-1);
letimate_FNR=fnr(1:end-1);
letimate_validFPR=~isnan(letimate_FPR);
letimate_validFNR=~isnan(letimate_FNR);
letimate_macroFPR = mean(letimate_FPR(letimate_validFPR));
letimate_macroFNR = mean(letimate_FNR(letimate_validFPR));
fprintf('\n (Macro FPR) = %.4f, (Macro FNR) = %.4f\n', ...
        letimate_macroFPR, letimate_macroFNR);


function [cov]=clac_coverage(Detector,Dr,num_Detector,featureNum,pexp, num_mc_points)
    cov=0;
    if num_Detector == 0
        cov = 0; 
        return;
    end
    
    for mc_idx = 1:num_mc_points 
       temp_c = rand(1,featureNum); 
       
       is_covered = false;
       for i=1:num_Detector
            dis = sqrt(sum((Detector(i,:)-temp_c).^2));
            if dis < Dr(i) 
               is_covered = true;
               break; 
            end
       end
       
       if is_covered
           cov = cov + 1;
       end
    end
    
    cov = cov / num_mc_points; 
end

