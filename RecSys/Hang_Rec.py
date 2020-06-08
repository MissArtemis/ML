import numpy as np

def Hang_Rec(rating_mat,u_mat,p_mat,k,maxCycles,lam1,lam2):
    m,n = np.shape(rating_mat)
    a = np.mat(np.random.random(size=(m,k)))
    b = np.mat(np.random.random(size=(k,n)))
    n1 = np.shape(u_mat)[1]
    n2 = np.shape(p_mat)[1]
    alpha = np.random.random(size=(m,n,n1))
    beta = np.random.random(size=(m,n,n2))
    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                error = rating_mat[i,j]
                for r1 in range(n1):
                    error = error - alpha[i,j,r1]*u_mat[i,r1]
                for r2 in range(n2):
                    error = error - beta[i,j,r2]*p_mat[j,r2]
                for r3 in range(k):
                    error = error - a[i,r3]*b[r3,j]
                for r1 in range(n1):
                    alpha[i,j,r1] = alpha[i,j,r1] - lam1*(2*error*(-u_mat[i,r1])+lam2*alpha[i,j,r1])
                for r2 in range(n2):
                    beta[i,j,r2] = beta[i,j,r2] - lam1*(2*error*(-p_mat[j,r2])+lam2*beta[i,j,r2])
                for r3 in range(k):
                    a[i,r3] = a[i,r3] - lam1*(2*error*(-b[r3,j])+lam2*a[i,r3])
                    b[r3,j] = b[r3,j] - lam1*(2*error*(-a[i,r3])+lam2*b[r3,j])
        loss = 0.0
        for i in range(m):
            for j in range(n):
                error = 0.0
                for r1 in range(n1):
                    error = error + alpha[i,j,r1]*u_mat[i,r1]
                for r2 in range(n2):
                    error = error + beta[i,j,r2]*p_mat[j,r2]
                for r3 in range(k):
                    error = error + a[i,r3]*b[r3,j]
                loss = (rating_mat[i,j]-error)**2
                for r1 in range(n1):
                    loss = loss + +lam2*alpha[i,j,r1]*alpha[i,j,r1]/2
                for r2 in range(n2):
                    loss = loss + +lam2*beta[i,j,r2]*beta[i,j,r2]/2
                for r3 in range(k):
                    loss = loss + +lam2 * (a[i,r3]*a[i,r3]+b[r3,j]*b[r3,j])/2
        if loss < 0.001:
            break
        if step%10 ==0:
            print("step:",step," loss:",loss)
    return a,b,alpha,beta


def gradAscent(dataMat, k, alpha, beta, maxCycles):

    m, n = np.shape(dataMat)
    p = np.mat(np.random.random(size=(m, k)))
    q = np.mat(np.random.random(size=(k, n)))

    # 2、开始训练
    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = dataMat[i, j]
                    for r in range(k):
                        error = error - p[i, r] * q[r, j]
                    for r in range(k):
                        # 梯度上升
                        p[i, r] = p[i, r] + alpha * (2 * error * q[r, j] - beta * p[i, r])
                        q[r, j] = q[r, j] + alpha * (2 * error * p[i, r] - beta * q[r, j])

        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = 0.0
                    for r in range(k):
                        error = error + p[i, r] * q[r, j]
                    # 3、计算损失函数
                    loss = (dataMat[i, j] - error) * (dataMat[i, j] - error)
                    for r in range(k):
                        loss = loss + beta * (p[i, r] * p[i, r] + q[r, j] * q[r, j]) / 2

        if loss < 0.001:
            break
        if step % 10 == 0:
            print("\titer: ", step, " loss: ", loss)
    return p, q


def prediction(a,b,alpha,beta,u_mat,p_mat,rating_mat,userid,prodid):
    rt = alpha[userid,prodid,:]*u_mat[userid,:].T + beta.swapaxes(0,1)[userid,prodid,:]*p_mat[prodid,:].T+a[userid,:]*b[:,prodid]
    #rt = beta.swapaxes(0,1)[userid,prodid,:]*p_mat[prodid,:].T+a[userid,:]*b[:,prodid]
    #rt = alpha[userid,prodid,:]*u_mat[userid,:].T+a[userid,:]*b[:,prodid]
    return rt


def pred_all(a,b,alpha,beta,u_mat,p_mat,rating_mat,userid):
    pred = {}
    pred_len = np.shape(p_mat)[1] 
    for prodid in range(pred_len):
        if rating_mat[userid,prodid-1] == 0:
            pred[prodid] = alpha[userid,prodid,:]*u_mat[userid,:].T + beta.swapaxes(0,1)[userid,prodid,:]*p_mat[prodid,:].T+a[userid,:]*b[:,prodid]
    return sorted(pred.items(), key=lambda d: d[1], reverse=True)


def top_k(pred,k):
    top_recom = []
    rs_len = len(pred)
    if k >= rs_len:
        top_recom = pred
    else:
        for i in range(k):
            top_recom.append(pred[i])
    return top_recom





if  __name__ == "__main__":
    print("Hello ZJH")
    #1 to 5 Rating, 4 products and 6 users
    # rating_mat = np.mat(np.random.random((6,4)))
    # print(rating_mat)
    # #Products matrix, 4 products 5 properties
    # p_mat = np.mat(np.random.random((4,5)))
    # print(p_mat)
    # #Users matrix, 6 users and 3 properties
    # u_mat = np.mat(np.random.random((6,3)))
    #print(u_mat)
    rating_mat=np.mat([[1,1,1,0,1],
                       [1,0,1,1,0],
                       [1,1,1,0,0],
                       [0,0,1,0,1],
                       [0,0,0,1,0],
                       [1,1,1,1,1]])
    print(rating_mat)
    p_mat =np.mat( [[1,1,0],
                    [1,0,1],
                    [1,1,1],
                    [1,0,0],
                    [0,0,1]])
    
    
    u_mat = np.mat([[1,1,1,0,0],
                    [1,1,1,1,1],
                    [1,0,0,0,1],
                    [1,1,1,1,0],
                    [1,0,1,0,1],
                    [0,1,0,1,0]])
    
    #a,b,alpha,beta = Hang_Rec(rating_mat,u_mat,p_mat,3,101,0.1,1)
    a,b,alpha,beta = Hang_Rec(rating_mat,u_mat,p_mat,3,101,0.1,1)
    p,q = gradAscent(rating_mat,3,0.1,1,101)
    print(prediction(a,b,alpha,beta,u_mat,p_mat,rating_mat,1,1))
    p_d = pred_all(a,b,alpha,beta,u_mat,p_mat,rating_mat,1)
    print(p_d)
    print(top_k(p_d,2))
        

    # print("a:",a)
    # print("b:",b)
    # print("alpha:",alpha)
    # print("beta:",beta)





















