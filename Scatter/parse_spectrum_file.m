function [spec] = parse_spectrum_file(filename)
%PARSE_SPECTRUM_FILE 此处显示有关此函数的摘要
%   此处显示详细说明
data = load(filename);
data(:,1) = data(:,1);
[N, ~] = size(data);
i_begin = 1;
while data(i_begin,2) == 0
    i_begin = i_begin+1;
end
i_end = i_begin;
while i_end <= N && data(i_end, 2) > 0
    i_end = i_end+1;
end
spec.num_bins_espc = i_end - i_begin + 1;
current_bin = spec.num_bins_espc;
spec.espc = zeros(200, 1, 'single');
spec.espc(1:current_bin) = data(i_begin:i_end, 1);
spec.espc(current_bin+1:end) = data(i_end,1);
spec.prob_espc_bin = zeros(200, 1, 'single');
spec.prob_espc_bin(1:current_bin) = data(i_begin:i_end, 2);

eng = 0.5*(spec.espc(1:current_bin) + spec.espc(2:current_bin+1));
spec.mean_energy_spectrum = sum(eng.*spec.prob_espc_bin(1:current_bin)) ...
    / sum(spec.prob_espc_bin(1:current_bin));
end

