
function [CFO_Offset,errorPhase_mean,errorMagnitude_mean,errorPhaseVar,errorMagnitudeVar,Refsymbols,eqSym_data] = data_process3(rx)

LongTrain = [1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1  ...  
     1 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1].'; 

DataSubcIdx = [7:11 13:25 27:32 34:39 41:53 55:59];
PilotSubcIdx = [12 26 40 54];


DataSubcPatt = [1:5 7:19 21:26 27:32 34:46 48:52]'; 
PilotSubcPatt = [6 20 33 47];   
UsedSubcIdx = [7:32 34:59];  

cfgHT = wlanNonHTConfig;
cfgHT.ChannelBandwidth = 'CBW20';
cfgHT.NumTransmitAntennas = 1;
cfgHT.MCS = 7;          % 64-QAM and 5/6 coding rate
fs = wlanSampleRate(cfgHT);           % Baseband sample rate

ind = wlanFieldIndices(cfgHT);

CFO_Offset = 0;
phase_est = 0;
cpe_data=0;
eqSym_data=0;
data_syms_out = 0;
 
DataSubcPatt = [1:5 7:19 21:26 27:32 34:46 48:52]';  
PilotSubcPatt = [6 20 33 47]; 
UsedSubcIdx = [7:32 34:59]; 


    tOff = wlanPacketDetect(rx,cfgHT.ChannelBandwidth);

    if(isempty(tOff)~=1)
 
        
   
                
                lstf = rx(tOff+(ind.LSTF(1):ind.LSTF(2)),:);
                coarseCFOEst = wlanCoarseCFOEstimate(lstf,cfgHT.ChannelBandwidth);
                rx = frequencyOffset(rx,fs,-coarseCFOEst);
                
                
                %Perform symbol timing synchronization.
                nonhtPreamble = rx(tOff+(ind.LSTF(1):ind.LSIG(2)),:);
                symOff = wlanSymbolTimingEstimate(nonhtPreamble,cfgHT.ChannelBandwidth);
                tOff = tOff+symOff;
                %tOff = 0;
                
                %Perform fine frequency offset correction.
                lltf = rx(tOff+(ind.LLTF(1):ind.LLTF(2)),:);
                fineCFOEst = wlanFineCFOEstimate(lltf,cfgHT.ChannelBandwidth);
                rx = frequencyOffset(rx,fs,-fineCFOEst);
                rx_signal=rx(tOff+(ind.LLTF(1):ind.NonHTData(2)),:);

                CFO_Offset = fineCFOEst+coarseCFOEst;
 
          
                %Perform channel estimation.
                    htltf = rx(tOff+(ind.LLTF(1):ind.LLTF(2)),:);
                    htltfDemod = wlanLLTFDemodulate(htltf,cfgHT);
                    chanEst= wlanLLTFChannelEstimate(htltfDemod,cfgHT);
                    
                    
                    %Perform noise estimation.
                    lstf = rx(tOff+(ind.LSTF(1):ind.LSTF(2)),:);
                    lltf = rx(tOff+(ind.LLTF(1):ind.LLTF(2)),:);
                    lltfDemod = wlanLLTFDemodulate(lltf,cfgHT);
                    noiseEst = wlanLLTFNoiseEstimate(lltfDemod);

                    
                    data=rx(tOff+(ind.NonHTData(1):ind.NonHTData(2)));
               %     [EVM,rmsEVM] = evmHT(data, chanEst, cfgHT);
                  
                    [rxPSDU_data,eqSym_data,cpe_data,peg] = wlanNonHTDataRecover(data,chanEst,noiseEst, cfgHT);
                    ttt = sum(sum(abs(eqSym_data)))/(48*38);



                    %%画星座图


                    

                   symbols = qamdemod(eqSym_data, 64,'UnitAveragePower', true);
                    Refsymbols = qammod(symbols,64,'UnitAveragePower', true);



                   scatterplot(eqSym_data(:), 1, 0, 'b.'); 

                    
                    errorMatrix = eqSym_data-Refsymbols;
                    evm = sqrt(sum(sum(abs(errorMatrix).^2)/sum(sum(abs(Refsymbols).^2))));
                    errorMagnitude = abs(eqSym_data)-abs(Refsymbols);
                    errorPhase = angle(eqSym_data)-angle(Refsymbols);
                    errorMagnitude_mean = sum(sum(abs(errorMagnitude)))/(38*48);
                    errorPhase_mean = sum(sum(abs(errorPhase)))/(38*48);
                    errorPhaseVar = var(reshape(errorPhase,1,38*48));
                    errorMagnitudeVar = var(reshape(errorMagnitude,1,38*48));
                    errorPhaseMax = max(reshape(errorPhase,1,38*48));
                    errorMagnitudeMax = max(reshape(errorMagnitude,1,38*48));
                    skewnessPhase = skewness(reshape(errorPhase,1,38*48)); 
                    skewnessMagnitude = skewness(reshape(errorMagnitude,1,38*48));
                    kurtosisPhase = kurtosis(reshape(errorPhase,1,38*48));
                    kurtosisMagnitude = kurtosis(reshape(errorMagnitude,1,38*48));
                    


                    Rxy = zeros(38,48);
                    for i =1:38  
                        x=eqSym_data(:,i)';
                        y=Refsymbols(:,i)';
                        N = length(x);  
                        y_conj = conj(y);  
                        
                        for tau = 0:(N-1)
                            Rxy(i,tau+1) = sum(x .* circshift(y_conj, [0, -tau]));
                        end
                        Rxy(i,:) = fft(Rxy(i,:));
                    end
                    rpsd = abs(sum(Rxy,1)/38);

    end

end
