% the code will be a little different from the method proposed by our paper

% we initialize the graph of each view and the unified graph outside the
% method, since we consider that our method is not restrcited to the way to
% building a graph, and the focal point of the proposed method is not how
% to build them.

% so we give you an interface which needs the constructed similarity matrix
% and the unified graph.

% in our code, we use k-nearest-neighbor (KNN) graph.

function [y]= OGIMC(Sor_ini,S_ini,num_cluster,numInst,lambda1,lambda2,lambda3,max_iter,ind_folds)

So = Sor_ini;
Sor = Sor_ini;
S = S_ini;

lambda = 1; 

% the lambda is the smooth parameter
% it will not change the clustering performace
% its role is to partition the data points into the required numbers
% In practice, we increase or decrease the value when the number of connected components is smaller or greater than c.

% % -------- Initialize the representation coefficient matrix B according to Sr---------- %
B = rand(length(Sor),length(Sor));   
% Initialize the reconstruction sparsity weight of all viewpoints to 1

B = B-diag(diag(B)); 
% Remove the influence of self representation

for m = 1:length(Sor)
    indx = [1:length(Sor)];
    indx(m) = [];
    B(indx',m) = (ProjSimplex(B(indx',m)'))'; 
    % Normalize each column of B by column
end

alpha = ones(length(Sor),1)/length(Sor);

% initialize alpha with the average value to every view

for iter = 1:max_iter
   

    % ---------- alpha_v ------------- %

    % solving problem (28) 
    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end   
    for iv = 1:length(Sor)
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        delta(1,iv) = -0.5*( norm((Sor{iv}-So{iv}).*W,'fro')^2+ lambda1*norm(vec_S(:,iv)-vec_S*B(:,iv))^2 + lambda2*norm(S-Sor{iv},'fro')^2 )/lambda3;
    end

    [alpha,~] = EProjSimplex_new(delta);
    % This problem can be solved by an efficient iterative algorithm proposed in [47].
    
    
    % ---------------- S ------------- %

    if iter == 1
       [y, S] = CLR(alpha,Sor,num_cluster,lambda); 
    else
       [y, S] = CLR(alpha,Sor,num_cluster,lambda,S); 
    end

    % ---------------- B---------------- %

    % Update variable B by solving (18);

    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end    

    for iv = 1:length(Sor)
        indv = [1:length(Sor)];
        indv(iv) = [];
        [B(indv',iv),~] = SimplexRepresentation_acc(vec_S(:,indv), vec_S(:,iv));
        %  min  || Bx - y||^2
        %  s.t. x>=0, 1'x=1
    end 
    
    % ------------------ Sr{iv} ------------------ %

    % Update variable S(v) via (14);
    
    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end
    
    for iv = 1:length(Sor)
        
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        M_iv = vec_S*B(:,iv);
        M_iv = reshape(M_iv,numInst,numInst);
        sum_Y = 0;
        coeef = 0;
        for iv2 = 1:length(Sor)
            if iv2 ~= iv
                Y_iv2 = vec_S(:,iv2)-vec_S*B(:,iv2)+B(iv,iv2)*vec_S(:,iv);
                sum_Y = sum_Y + alpha(iv2)*B(iv,iv2)*lambda1*Y_iv2;
                coeef = coeef +  B(iv,iv2)^2*alpha(iv2);
                clear Y_iv2
            end
        end
        clear iv2
        matrix_sum_Y = reshape(sum_Y,numInst,numInst);
        clear sum_Y
        Linshi_L = (alpha(iv)*Sor{iv}.*W+  alpha(iv)*lambda1*M_iv  +alpha(iv)*lambda2*length(Sor)*S  +matrix_sum_Y)./(alpha(iv)*W+lambda1*alpha(iv)+coeef*lambda1);
        for num = 1:numInst
            indnum = [1:numInst];
            indnum(num) = [];
            Sor{iv}(indnum',num) = (EProjSimplex_new(Linshi_L(indnum',num)'))';
        end
        clear Linshi_L matrix_sum_Y coeef
    end
    clear vec_S 
    
    % --------------  obj -------------- %

    % compute obj function value
    % if needed, you can add the obj into the return list

    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end
    for iv = 1:length(Sor)
        % ------- obj reconstructed error ------------ %
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        linshi_S = 0.5*(Sor{iv}+Sor{iv}');
        LSv = diag(sum(linshi_S))-linshi_S;
        Rec_error(iv) = norm((Sor{iv}-So{iv}).*W,'fro')^2+lambda2*norm(S-Sor{iv},'fro')^2+lambda1*norm(vec_S(:,iv)-vec_S*B(:,iv))^2;   
    end
    clear W ind_0 linshi_S LSv

    obj(iter) = alpha*Rec_error'+lambda3*norm(alpha,2)^2;

    
   
    clear vec_S
    
    if (iter > 2 && abs(obj(iter)-obj(iter-1))<1e-5)   
        iter
        break;
    end
    % we suppose that if the absolute value gap between two iteration is
    % less than 1e-5, then our method converges.

    % in real IMVC task, the terminating condition shoule change with the
    % value of hyperparameters.

    end
end
