function [m_database V_PCA  ProjectedImages_PCA] = PCACore(T,classinfo)
% Use Principle Component Analysis (PCA) and Fisher Linear Discriminant (FLD) to determine the most 
% discriminating features between images of faces.
%
% Description: This function gets a 2D matrix, containing all training image vectors
% and returns 4 outputs which are extracted from training database.

% Suppose Ti is a training image, which has been reshaped into a 1D vector.
% Also, P is the total number of MxN training images and C is the number of
% classes. At first, centered Ti is mapped onto a (P-C) linear subspace by V_PCA

%   P=training image 갯수, C=class 갯수
%   Ti= training image를 1D vector로 reshape한거
%   Ti를 normalizing하고 PCA를 이용해서 P-C개의 linear subspce로 projection시킴(그게 Zi)

% transfer matrix: Zi = V_PCA * (Ti - m_database).
% Then, Zi is converted to Yi by projecting onto a (C-1) linear subspace, so that 
% images of the same class (or person) move closer together and images of difference 
% classes move further apart: Yi = V_Fisher' * Zi = V_Fisher' * V_PCA' * (Ti - m_database)

% Zi를 FLD를 이용해서 C-1개의 linear subspace로 projection시킴

%
% Argument:      T                      - (M*NxP) A 2D matrix, containing all 1D image vectors.
%                                         All of 1D column vectors have the same length of M*N 
%                                         and 'T' will be a MNxP 2D matrix.
% 
% Returns:       m_database             - (M*Nx1) Mean of the training database
%                V_PCA                  - (M*Nx(P-C)) Eigen vectors of the covariance matrix of the 
%                                         training database
%                V_Fisher               - ((P-C)x(C-1)) Largest (C-1) eigen vectors of matrix J = inv(Sw) * Sb
%                ProjectedImages_Fisher - ((C-1)xP) Training images, which are projected onto Fisher linear space
%
% See also: EIG

% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir                  


% Class_number = ( size(T,2) )/2; % Number of classes (or persons)
Class_number = length(classinfo);
% Class_population = 2; % Number of images in each class
Class_population = classinfo(:,2);
% P = Class_population * Class_number; % Total number of training images
P = size(T,2);

%%%%%%%%classinformation을 누적치로 만듬
accumclass_pop=Class_population;
for t=1:length(Class_population)-1
    accumclass_pop(t+1) = accumclass_pop(t) + accumclass_pop(t+1);
end
tic
%%%%%%%%%%%%%%%%%%%%%%%% calculating the mean image 
disp('PCA 시작!')
m_database = mean(T,2); %평균값

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the deviation of each image from mean image
A = T - repmat(m_database,1,P);

%%%%%%%%%%%%%%%%%%%%%%%% Snapshot method of Eigenface algorithm
L = A'*A; % L is the surrogate of covariance matrix C=A*A'. covariance matrix의 대체
[V D] = eig(L); % Diagonal elements of D are the eigenvalues for both L=A'*A and C=A*A'.

%%%%%%%%%%%%%%%%%%%%%%%% Sorting and eliminating small eigenvalues
% L_eig_vec = [];수정해버림
% % for i = 1 : P-Class_number 
% %     L_eig_vec = [L_eig_vec V(:,i)]; %P-C개의 eigenvector 추출
% % end

L_eig_vec=V(:,1:P-Class_number);
%%%%%%%%%%%%%%%%%%%%%%%% Calculating the eigenvectors of covariance matrix 'C'
V_PCA = A * L_eig_vec; % A: centered image vectors

%%%%%%%%%%%%%%%%%%%%%%%% Projecting centered image vectors onto eigenspace
% Zi = V_PCA' * (Ti-m_database)         eigenvector로 embedding

% % ProjectedImages_PCA = [];수정해버림

% % for i = 1 : P수정해버림
% %     temp = V_PCA'*A(:,i);수정해버림
% %     ProjectedImages_PCA = [ProjectedImages_PCA temp]; 수정해버림
% % end수정해버림

ProjectedImages_PCA = V_PCA'*A(:,1:P);
disp('PCA_complete')
toc

end




