function [V_Fisher ProjectedImages_Fisher] = FLDcore(classinfo,ProjectedImages_PCA,Sb,Sw)

Class_number = length(classinfo);
% Class_population = 2; % Number of images in each class
Class_population = classinfo(:,2);
% P = Class_population * Class_number; % Total number of training images


%%%%%%%%classinformation을 누적치로 만듬



%%%%%%%%%%%%%%%%%%%%%%%% Calculating Fisher discriminant basis's
% We want to maximise the Between Scatter Matrix, while minimising the
% Within Scatter Matrix. Thus, a cost function J is defined, so that this condition is satisfied.
tic
[J_eig_vec, J_eig_val] = eig(Sb,Sw); % Cost function J = inv(Sw) * Sb
J_eig_vec = fliplr(J_eig_vec);

%%%%%%%%%%%%%%%%%%%%%%%% Eliminating zero eigens and sorting in descend order
% % for i = 1 : Class_number-1 
% %     V_Fisher(:,i) = J_eig_vec(:,i); % Largest (C-1) eigen vectors of matrix J
% % end

V_Fisher=J_eig_vec(:,1:Class_number-1);
%%%%%%%%%%%%%%%%%%%%%%%% Projecting images onto Fisher linear space
% Yi = V_Fisher' * V_PCA' * (Ti - m_database) 
% % for i = 1 : Class_number*Class_population
% %     ProjectedImages_Fisher(:,i) = V_Fisher' * ProjectedImages_PCA(:,i);
% % end
ProjectedImages_Fisher=V_Fisher'*ProjectedImages_PCA(:,1:Class_number*Class_population);
disp('everybody_complete')
toc


end