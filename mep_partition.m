function xpmin = mep_partition(y, NS)
% code written by Najah Ghalyan
    NP=NS-1;
    x1 = sort(y);
    T=size(y, 2); 
    k=round(T/NS);
    kp=1;
    for p=1:NP 
        kp=kp+k;
        xpmin(p)=x1(kp);
    end
