function [MCparam] = load_MCGPU_param(filepath)
%LOAD_MCGPU_PARAM parse .in files for MCGPU
%   此处显示详细说明
fid = fopen(filepath, 'r');
new_line = fgetl(fid);
while ~contains(new_line, "SECTION SIMULATION CONFIG")
    new_line = fgetl(fid);
end
%% simulation configurations
MCparam.total_histories = sscanf(fgetl_trimmed(fid), '%lf')+0.0001;
MCparam.seed_input = sscanf(fgetl_trimmed(fid), '%d');
MCparam.gpu_id = sscanf(fgetl_trimmed(fid), '%d');
MCparam.num_threads_per_block = sscanf(fgetl_trimmed(fid), '%d');
if mod(MCparam.num_threads_per_block, 32) ~= 0
    error("!!read_input ERROR!! The input number of GPU threads per CUDA block ..." + ...
        "must be a multiple of 32 (warp size). Input value: %d !!\n", ...
        MCparam.num_threads_per_block);
end
MCparam.histories_per_thread = sscanf(fgetl_trimmed(fid), '%d');

new_line = fgetl(fid);
while ~contains(new_line, "SECTION SOURCE")
    new_line = fgetl(fid);
end
%% source parameters
MCparam.file_name_espc = sscanf(fgetl_trimmed(fid), '%s');
try
    MCparam.spec = parse_spec_file(MCparam.file_name_espc);
catch
    disp('spec file not found, please reload in your script!');
end
MCparam.source_pos = sscanf(fgetl_trimmed(fid), '%f %f %f');
MCparam.source_dir = sscanf(fgetl_trimmed(fid), '%f %f %f');
MCparam.source_dir = MCparam.source_dir / norm(MCparam.source_dir);
MCparam.aperture = sscanf(fgetl_trimmed(fid), '%f %f');
if MCparam.aperture(1) > 180.0
    error("!!read_input ERROR!! Input polar aperture must be in [0,180] deg.!\n");
end
if MCparam.aperture(2) > 360.0
    error("!!read_input ERROR!! Input azimuthal aperture must be in [0,360] deg.!\n");
end

new_line = fgetl(fid);
while ~contains(new_line, "SECTION IMAGE DETECTOR")
    new_line = fgetl(fid);
end
%% output & detector parameters
MCparam.output_proj_type = sscanf(fgetl_trimmed(fid), '%d');
MCparam.file_name_output = sscanf(fgetl_trimmed(fid), '%s');
MCparam.dummy_num_pixels = sscanf(fgetl_trimmed(fid), '%f %f');
MCparam.dummy_num_pixels = int32(MCparam.dummy_num_pixels+0.001);
MCparam.total_num_pixels = prod(MCparam.dummy_num_pixels);
if MCparam.total_num_pixels < 1 || MCparam.total_num_pixels > 99999999
    error("!!read_input ERROR!! The input number of pixels is incorrect. Input: ..." + ...
        "total_num_pix = %d!!\n", MCparam.total_num_pixels);
end
MCparam.det_size = sscanf(fgetl_trimmed(fid), '%f %f');
MCparam.sdd = sscanf(fgetl_trimmed(fid), '%f');
if MCparam.sdd < 1e-6
    error("!!read_input ERROR!! The source-to-detector distance must be positive.\n");
end

new_line = fgetl(fid);
while ~contains(new_line, "SECTION CT SCAN TRAJECTORY")
    new_line = fgetl(fid);
end
%% CT scan parameters
MCparam.num_projections = sscanf(fgetl_trimmed(fid), '%d');
if MCparam.num_projections < 1
    MCparam.num_projections = 1;
end
if MCparam.num_projections ~= 1
    MCparam.D_angle = sscanf(fgetl_trimmed(fid), '%lf');
    MCparam.angularROI = sscanf(fgetl_trimmed(fid), '%lf %lf');
    MCparam.sod = sscanf(fgetl_trimmed(fid), '%lf');
    if MCparam.sod < 0.0 || MCparam.sod > MCparam.sdd
        error("!!read_input ERROR!! Invalid source-to-rotation axis distance!\n");
    end
    MCparam.vertical_translation_per_projection = sscanf(fgetl_trimmed(fid), '%lf');
end

new_line = fgetl(fid);
while ~contains(new_line, "SECTION DOSE DEPOSITION")
    new_line = fgetl(fid);
end
%% dose deposition
line = fgetl_trimmed(fid);
if strncmpi(line, 'ye', 2)
    MCparam.flag_material_dose = 1;
else
    MCparam.flag_material_dose = 0;
end
line = fgetl_trimmed(fid);
if strncmpi(line, 'ye', 2)
    MCparam.tally_3D_dose = 1;
else
    MCparam.tally_3D_dose = 0;
end
if MCparam.tally_3D_dose
    MCparam.file_dose_output = sscanf(fgetl_trimmed(fid), '%s');
    MCparam.dose_ROI_x = sscanf(fgetl_trimmed(fid), '%f %f') - 1;
    MCparam.dose_ROI_y = sscanf(fgetl_trimmed(fid), '%f %f') - 1;
    MCparam.dose_ROI_z = sscanf(fgetl_trimmed(fid), '%f %f') - 1;
end

new_line = fgetl(fid);
while ~contains(new_line, "SECTION VOXELIZED GEOMETRY FILE v.2009-11-30")
    new_line = fgetl(fid);
end
%% voxel file
MCparam.file_name_voxels = sscanf(fgetl_trimmed(fid), '%s');

new_line = fgetl(fid);
while ~contains(new_line, "SECTION MATERIAL")
    new_line = fgetl(fid);
end
%% materials
for i = 1:20
    line = fgetl_trimmed(fid);
    if line == -1
        MCparam.file_name_materials{i} = '\n';
    else
        MCparam.file_name_materials{i} = line;
    end
end
fclose(fid);
end

function spec = parse_spec_file(fname)
fid = fopen(fname, 'r');
current_bin = 0;
prob = 100;
spec.prob_espc_bin = zeros(200, 1, 'single');
spec.espc = zeros(200, 1, 'single');
while prob > -1e-11
    current_bin = current_bin + 1;
    if current_bin > 199
        error("!!init_energy_spectrum ERROR!!: too many energy bins in the input spectrum. \n");
    end
    prob = -123456789.0;
    dat = sscanf(fgetl_trimmed(fid), '%f %f');
    prob = dat(2);
    spec.prob_espc_bin(current_bin) = prob;
    spec.espc(current_bin) = dat(1);
    if prob == -123456789.0
        error("!!init_energy_spectrum ERROR!!: invalid energy bin number!\n");
    elseif current_bin > 1 && dat(1) < spec.espc(current_bin-1)
        error("!!init_energy_spectrum ERROR!!: input energy bins with decreasing energy?\n");
    end
end
spec.num_bins_espc = current_bin;
spec.espc(current_bin+1:end) = spec.espc(current_bin);
spec.prob_espc_bin(current_bin) = 0;
eng = 0.5*(spec.espc(1:current_bin) + spec.espc(2:current_bin+1));
spec.mean_energy_spectrum = sum(eng.*spec.prob_espc_bin(1:current_bin)) ...
    / sum(spec.prob_espc_bin(1:current_bin));
fclose(fid);
end

function line = fgetl_trimmed(fid)
% remove spaces and comments
line = fgetl(fid);
if line == -1
    return;
end
line = char(line);
if ~isempty(find(line=='#', 1))
    line(find(line=='#', 1):end) = [];
end
end