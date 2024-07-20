function [voxel_data] = parse_vox_file(filepath)
%PARSE_VOX_FILE 此处显示有关此函数的摘要
%   此处显示详细说明
fid = fopen(filepath, 'r');
new_line = fgetl(fid);
while ~contains(new_line, "SECTION VOXELS")
    new_line = fgetl(fid);
end
voxel_data.num_voxels = sscanf(fgetl(fid), '%f %f %f');
voxel_data.voxel_size = sscanf(fgetl(fid), '%f %f %f');
fclose(fid);
[voxel_data.material, voxel_data.density, voxel_data.density_max] = parse_vox_mex(filepath, voxel_data.num_voxels, voxel_data.voxel_size);
end

