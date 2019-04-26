function Sd = integralImgSqDiff(PaddedImg,Ds,t1,t2)  
Dist2=(PaddedImg(1+Ds:end-Ds,1+Ds:end-Ds)-PaddedImg(1+Ds+t1:end-Ds+t1,1+Ds+t2:end-Ds+t2)).^2;  
Sd = cumsum(Dist2,1);  
Sd = cumsum(Sd,2);