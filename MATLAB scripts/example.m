% A sample script, which shows the usage of functions, included in
% FLD-based face recognition system (Fisherface method)
%
% See also: CREATEDATABASE, FISHERFACECORE, RECOGNITION

% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir                  

clear all
clc
close all

%%
% You can customize and fix initial directory paths
TrainDatabasePath = uigetdir(strcat(matlabroot,'\work'), 'Select training database path' );
%%
TestDatabasePath = uigetdir(strcat(matlabroot,'\work'), 'Select test database path');

prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of FLD-Based Face Recognition System';
num_lines= 1;
def = {'1'};

TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.png');
im = imread(TestImage);
%%
T = CreateDatabase(TrainDatabasePath);
%%
% flddatainf.csv 가져와야함//행렬로!!!
classinfo=flddatainfo;
%% PCA training
[m_database V_PCA  ProjectedImages_PCA] = PCACore(T,classinfo);
%% Fisherface training
[Sb Sw] = Fisherface(T,classinfo,ProjectedImages_PCA);

%% FLD

[V_Fisher ProjectedImages_Fisher] = FLDcore(classinfo,ProjectedImages_PCA,Sb,Sw);

%% or total training
[m_database V_PCA V_Fisher ProjectedImages_Fisher] = PCAandFisherface(T,flddatainfo)

%%  test
OutputName = Recognition(TestImage, m, V_PCA, V_Fisher, ProjectedImages_Fisher);

SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);
subplot(1,2,1)
imshow(im)
title('Test Image');
subplot(1,2,2)
imshow(SelectedImage);
title('Equivalent Image');

str = strcat('Matched image is :  ',OutputName);
disp(str)
