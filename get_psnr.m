function result = get_psnr( image1, image2 )
[h, w] = size(image1);
B = 8;
MAX = 2 ^ B - 1;
MSE = sum(sum((image1 - image2) .^ 2)) / (h * w);
result = 20 * log10(MAX / sqrt(MSE));
end

