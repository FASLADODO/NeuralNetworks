function  [G1 , G2 , G3 ] = bayes_discriminant(x1,x2,x3,data1,data2, data3)

% Covariance
covar1=cov(data1);
covar2=cov(data2);
covar3=cov(data3);

% Mean
m1=mean(data1);
m2=mean(data2);
m3=mean(data3);

% Determinant of the covariance matrix
detcov1=det(covar1);
detcov2=det(covar2);
detcov3=det(covar3);

mat1 = [x1-m1(1) x2-m1(2)];
mat2 = [x1-m2(1) x2-m2(2)];
mat3 = [x1-m3(1) x2-m3(2)];

o1=-0.5*mat1*(covar1\mat1')-0.5*log(detcov1)-log(2* pi);
o2=-0.5*mat2*(covar2\mat2')-0.5*log(detcov2)-log(2* pi);
o3=-0.5*mat3*(covar3\mat3')-0.5*log(detcov3)-log(2* pi);
%%%% Discriminant function
eq2 = simplify(o2);
eq1 = simplify(o1);
eq3 = simplify(o3);
%%%%Decision Boundary 
G1 = simplify(eq1 - eq2 );
G2 = simplify(eq2 - eq3 );
G3 = simplify(eq3 - eq1 );

end