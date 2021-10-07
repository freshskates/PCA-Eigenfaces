clear, clc, close all

% PCA Eigenfaces
% settings 
global img_width
global img_height

img_width = 32
img_height = 32

% Load training data
% train_data = loadData("./ALL/*.TIF", sample_size)
train_data = loadData("./FA/*.TIF", -1)
test_data = loadData("./FB/*.TIF", -1)

plotRawImages(train_data, -1, "train ")
plotRawImages(test_data, -1, "test ")

% get average face
mean_img = getMeanImage(train_data)

% Center data 
% image - mean_img 
centerData(train_data, mean_img)
 
% Display the average face of the dataset
figure('Name','Average Face'); 
imshow(reshape(mean_img, [img_width, img_height]), []);

% Covariance Matrix
% cov_mat = D * D' (transposed)
cov_mat = covarianceMatrix(train_data)

% Eigen Values and Vectors
% [V, D] = eig(Data)
% V: eigenvector
% D: diagonal of the Data
%
% Eigen vectors were normalized 
% Formula: V[i] / sqrt(eigValues[i] 
[eigenFaces, eigValues] = getEigenFaces(cov_mat, train_data)

% Display linear chart of eigen values and faces
plotEigenLineChart(eigValues)
plotEigenFaces(eigenFaces, train_data, -1)

%% eigenface detection-training image
% proj_train=zeros(img_width*img_height, sample_size, 10);
% all_faces=zeros(img_width*img_height, sample_size, 10);
% for i=1:5
%     for j=1:sample_size

%% my utility functions 
% - loadData -> 
% - getMeanImage -> 
% - centerData ->
% - covarianceMatrix ->
% - getEigenFaces -> 
% - plotEigenLineChart -> void
% - plotEigenFaces -> void
%% 
function train_data = loadData(img_path, quantity)
    global img_width
    global img_height

    train_data = [];
    data = imageDatastore(img_path)
    if quantity == -1
        quantity = length(data.Files); 
    end

    temp_train_data = zeros(quantity, img_width * img_height);
    for i = 1:quantity
        train_data_temp_1 = imread(data.Files{i})
        temp_train_data(i,:) = reshape(train_data_temp_1, [1, img_width*img_height]);
    end
    
    train_data = [train_data; temp_train_data];
    
    train_data = train_data'
end

function mean_img = getMeanImage(data)
    mean_img = mean(data, 2)
end

function [] = centerData(data, mean_img)
    for i = 1:size(data, 2)
        data(:, i) = data(:, i) - mean_img
    end
end

function cov_mat = covarianceMatrix(data)
    cov_mat = data' * data;
end

function [eigenFaces, eigValues] = getEigenFaces(covariance_mat, data)
    [V, D] = eig(covariance_mat);
    eigValues = diag(D);

    eigValues = eigValues(end:-1:1);
    V = V(:,end:-1:1);
    
    %% normalizing eigen vector
    % resource that helped me
    % https://physics.stackexchange.com/questions/600187/normalizing-eigenvectors
    for i = 1:size(V, 2)
        V(:,i) = V(:, i)./sqrt(eigValues(i));
    end
    
    eigenFaces = data * V;
end

function [] = plotEigenLineChart(eigValues)
    figure('Name','Eigen Values'); 
    xlabel('indexes'); 
    ylabel('eigen values'); 
    plot(eigValues);
end

function [] = plotEigenFaces(eigenFaces, data, quantity)
    global img_width
    global img_height

    [W, H] = size(data)
    if quantity == -1
        quantity = H
    end        

    H = ceil(H/4)

    i = 1;
    figure("Name", "Eigen Faces Plot");
    for numOfEig = 1:quantity
        eigFaces_sub = eigenFaces(:, i:numOfEig);
        wt = data(:, i)' * eigFaces_sub; % weighting
        vi = reshape(eigFaces_sub * wt', img_width, img_height); % projection
        subplot(4, H, i);
        imshow(vi, []); 
    
        % title([num2str(numOfEig), ' eigen faces']);
        i = i + 1;
    end
end

function [] = plotRawImages(data, quantity, header)
    global img_width
    global img_height

    [W, H] = size(data)
    if quantity == -1
        quantity = H
    end        
    % to make code scaleable when plotting - H = ceil(H/4)
    H = ceil(H/4)

    figure("Name", strcat("Data: ", header));
    for i = 1:quantity
        vi = reshape(data(:, i), img_width, img_height); % projection
        subplot(4, H, i);
        imshow(vi, []); 
        title([header, num2str(i)]);
    end
end
