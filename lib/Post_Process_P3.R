if(!require("pracma")){
  install.packages("pracma")
}
library(pracma)

K=function(x_i_t,x_j_t){
  return (exp(2*(x_i_t%*%t(x_j_t)-1)))
}

X_mat=function(q){
  X=apply(t(q), 2 ,function(x) x/Norm(x))
  return (X)
}

svd_krr=function(n,lambda=0.5,X,y){
  I=diag(n)
  res=K(X,X)%*%solve(K(X,X)+lambda*I)%*%y
  return (res)
}