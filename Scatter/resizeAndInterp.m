function [interp_projs] = resizeAndInterp(projs, reH, reW, num_views)
%RESIZE 此处显示有关此函数的摘要
%   此处显示详细说明
[~,~,views] = size(projs);
new_projs = zeros(reW, reH, views);
for idx = 1:views
    new_projs(:,:,idx) = imresize(projs(:,:,idx), [reW, reH]);
end
sinos = permute(new_projs, [1,3,2]);
x = 1:size(sinos, 2);
xq = linspace(1, size(sinos, 2), num_views);
interp_sinos = zeros(size(sinos, 1), num_views, size(sinos, 3));
for i = 1:size(sinos, 3)
    interp_sinos(:,:,i) = interp1(x, squeeze(sinos(:, :, i))', xq, 'spline', 'extrap')';
end
interp_projs = permute(interp_sinos, [1 3 2]);
end

