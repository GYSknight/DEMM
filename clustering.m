function [A,W,Z,G,F,iter,obj,alpha] = algo_OMSC(X,Y,lambda,d,numanchor)
maxIter = 50 ; 
m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);
W = cell(numview,1);    
A = zeros(d,m);         
Z = zeros(m,numsample); 
for i = 1:numview
   di = size(X{i},1); 
   W{i} = zeros(di,d);
   X{i} = mapstd(X{i}',0,1); % turn into d*n
end
Z(:,1:m) = eye(m);
G = eye(m,numclass);
F = eye(numclass,numsample); 
alpha = ones(1,numview)/numview;
opt.disp = 0;
flag = 1;
iter = 0;
while flag
    iter = iter + 1;
    AZ = A*Z; 
    for iv = 1:numview
        C = gpuArray(X{iv}) * gpuArray(AZ');
        [U,~,V] = svd(C,'econ');
        W{iv} = gather(U * V');
    end
    sumAlpha = 0;
    part1 = 0;
    for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = part1 + al2 * W{ia}' * X{ia} * Z';
    end
    [Unew,~,Vnew] = svd(part1,'econ');
    A = Unew*Vnew';
    H = 2*sumAlpha*eye(m)+2*lambda*eye(m);
    H = (H+H')/2;
    options = optimset('Algorithm','interior-point-convex','Display','off'); % interior-point-convex
    parfor ji=1:numsample
        ff=0;
        e = F(:,ji)'*G'; % 计算 e
        for j=1:numview
            C = W{j} * A;
            ff = ff - 2*X{j}(:,ji)'*C - 2*lambda*e;
        end
        Z(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
    end
    J = Z*F';      
    [Ug,~,Vg] = svd(J,'econ');
    G = Ug*Vg';
    
    %% using cosine distance
    F=zeros(numclass,numsample);
    for i=1:numsample
        Dis=zeros(numclass,1);
        for j=1:numclass
            numerator = dot(Z(:,i), G(:,j));
            denominator = (norm(Z(:,i),2) * norm(G(:,j),2));
            if denominator == 0
                Dis(j) = 1; % Maximal distance if denominator is zero
            else
                Dis(j) = 1 - (numerator / denominator);
            end
        end
        [~,r]=min(Dis);
        F(r(1),i)=1;
    end
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm(X{iv} - W{iv} * A * Z,'fro')^2;
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;


    term1 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - W{iv} * A * Z,'fro')^2;
    end
    term2 = lambda * norm(Z - G * F,'fro')^2; 
    obj(iter) = term1 + term2;
    
    if (iter==15) 
        flag = 0;
    end
end
 
         
    
