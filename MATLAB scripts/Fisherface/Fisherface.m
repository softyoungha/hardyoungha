function [Sb Sw] = Fisherface(T,classinfo,ProjectedImages_PCA)

Class_number = length(classinfo);
% Class_population = 2; % Number of images in each class
Class_population = classinfo(:,2);
% P = Class_population * Class_number; % Total number of training images
P = size(T,2);
m_database = mean(T,2);
%%%%%%%%classinformation을 누적치로 만듬
accumclass_pop=Class_population;

tic
m_PCA = mean(ProjectedImages_PCA,2); % Total mean in eigenspace
m = zeros(P-Class_number,Class_number); %각 class의 평균 저장함
Sw = zeros(P-Class_number,P-Class_number); % Initialization os Within Scatter Matrix
Sb = zeros(P-Class_number,P-Class_number); % Initialization of Between Scatter Matrix

%Sw와 Sb 계산
for i = 1 : Class_number
    
%     m(:,i) = mean(  ProjectedImages_PCA(:,((i-1)*Class_population+1):i*Class_population) , 2 )';    %%%%%%%%%%
    m(:,i) = mean( ProjectedImages_PCA(:,accumclass_pop(i):(accumclass_pop(min(i+1,Class_number))-1)) , 2);
    
    S  = zeros(P-Class_number,P-Class_number); 
%     for j = ( (i-1)*Class_population+1 ) : ( i*Class_population )
%         S = S + (ProjectedImages_PCA(:,j)-m(:,i))*(ProjectedImages_PCA(:,j)-m(:,i))';
%     end
    for j =  accumclass_pop(i): (accumclass_pop(min(i+1,Class_number))-1)
        S = S + (ProjectedImages_PCA(:,j)-m(:,i))*(ProjectedImages_PCA(:,j)-m(:,i))';
    end


    Sw = Sw + S; % Within Scatter Matrix
    
    Sb = Sb + Class_population(i) * (m(:,i)-m_PCA) * (m(:,i)-m_PCA)'; % Between Scatter Matrix
    if mod(i,500)==0
        i
    end
end
disp('SwSb_complete')
toc


end