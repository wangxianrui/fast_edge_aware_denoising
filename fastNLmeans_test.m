clc
clear all
ima=double(imread('standard_test_images/lena_gray_512.tif'));
[height, width, channel] = size(ima);

% add  noise
sigma=10;
rima=ima+sigma*randn(size(ima)); 

% denoise
fima=fastNLmeans(rima, 7, 13, sigma);

get_psnr(ima, rima)
get_psnr(ima, fima)
% show results
subplot(1,3,1),imshow(uint8(ima)),title('original');
subplot(1,3,2),imshow(uint8(rima)),title('noisy');
subplot(1,3,3),imshow(uint8(fima)),title('filtered');
