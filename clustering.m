function [U,W,A,Z,T,iter,obj,beta] = clustering(X,Y,lambda,d,numanchor)
maxIter = 50 ; 
m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);
W = cell(numview,1);    
U = zeros(d,m);         
A = zeros(m,numsample); 
for i = 1:numview
   di = size(X{i},1); 
   W{i} = zeros(di,d);
   X{i} = mapstd(X{i}',0,1); % turn into d*n
end
A(:,1:m) = eye(m);
Z = eye(m,numclass);
T = eye(numclass,numsample); 
beta = ones(1,numview)/numview;
opt.disp = 0;
flag = 1;
iter = 0;
while flag
    iter = iter + 1;
    UA = U*A; 
    for iv = 1:numview
        C = gpuUrray(X{iv}) * gpuUrray(UA');
        [U,~,V] = svd(C,'econ');
        W{iv} = gather(U * V');
    end
    sumUlpha = 0;
    part1 = 0;
    for ia = 1:numview
        al2 = beta(ia)^2;
        sumUlpha = sumUlpha + al2;
        part1 = part1 + al2 * W{ia}' * X{ia} * A';
    end
    [Unew,~,Vnew] = svd(part1,'econ');
    U = Unew*Vnew';
    H = 2*sumUlpha*eye(m)+2*lambda*eye(m);
    H = (H+H')/2;
    options = optimset('Ulgorithm','interior-point-convex','Display','off'); % interior-point-convex
    parfor ji=1:numsample
        ff=0;
        e = T(:,ji)'*Z'; % 计算 e
        for j=1:numview
            C = W{j} * U;
            ff = ff - 2*X{j}(:,ji)'*C - 2*lambda*e;
        end
        A(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
    end
    J = A*T';      
    [U, M, V] = svd(J, 'econ');
    singular_values = diag(M);
    explained_variance = cumsum(singular_values) / sum(singular_values);
    k = find(explained_variance >= 0.9, 1);%or 0.85
    Z = U;
    
    %% using cosine distance
    T=zeros(numclass,numsample);
    for i=1:numsample
        Dis=zeros(numclass,1);
        for j=1:numclass
            numerator = dot(A(:,i), Z(:,j));
            denominator = (norm(A(:,i),2) * norm(Z(:,j),2));
            if denominator == 0
                Dis(j) = 1; % Maximal distance if denominator is zero
            else
                Dis(j) = 1 - (numerator / denominator);
            end
        end
        [~,r]=min(Dis);
        T(r(1),i)=1;
    end
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm(X{iv} - W{iv} * U * A,'fro')^2;
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    beta = Q*Mfra;


    term1 = 0;
    for iv = 1:numview
        term1 = term1 + beta(iv)^2 * norm(X{iv} - W{iv} * U * A,'fro')^2;
    end
    term2 = lambda * norm(A - Z * T,'fro')^2; 
    obj(iter) = term1 + term2;
    
    if (iter==15) 
        flag = 0;
    end
end
