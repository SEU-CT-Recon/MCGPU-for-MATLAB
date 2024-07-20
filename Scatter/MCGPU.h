/*
 * @Author: Tianling Lyu
 * @Date: 2022-08-26 15:33:58
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2023-12-30 23:23:12
 * @FilePath: \firstRECON2_matlab\MATLAB\Scatter\MCGPU.h
 */

#pragma comment(linker, "/STACK:102400000,102400000")

#ifndef MCGPU2_H_
#define MCGPU2_H_

#include "MATLAB/Scatter/common_defines.h"
#include <mex.h>
#include "singleton.h"

class MCGPU : public Singleton<MCGPU>
{
public:
    // Ctor and Dtor
    MCGPU();
    ~MCGPU();
    // utility functions
    bool initialize(mxArray* params);
    bool run(mxArray* params, float* noScatter, float* compton, float* rayleigh, 
		float* multiscatter);
    bool clear();

	size_t getProjWidth();
	size_t getProjHeight();
	size_t getProjNum();
private:
	bool parse_param_struct(mxArray* params);
	bool parse_spec_struct(mxArray* params);
	bool parse_data_struct(mxArray* params);
    //// *** HOST FUNCTIONS *** ////
    bool set_CT_trajectory(int num_projections, double D_angle, 
        double angularROI_0, double angularROI_1, double SRotAxisD, 
        source_struct* source_data, detector_struct* detector_data, 
        double vertical_translation_per_projection);
    bool init_energy_spectrum(char* file_name_espc, 
        source_energy_struct* source_energy_data, float *mean_energy_spectrum);
    void update_seed_PRNG(int batch_number, unsigned long long total_histories, 
        int* seed);
    void IRND0(float *W, float *F, short int *K, int N);
    bool _track_particles(dim3 block, dim3 thread, int num_p);
	bool report_image_to_array(float* noScatter, float* compton, float* rayleigh, 
		float* multiscatter);

    // print functions
    void _print_init();
    void _print_read();
    void _print_allocated();
    bool _check_consistent();
    void _print_mc_start(time_t clock_start);
    bool _check_angle(double current_angle, int num_p);
    void _print_finish(clock_t clock_start_beginning, double time_total_MC_simulation);
    void _check_block(int &total_threads, int &total_threads_blocks);

    // *** Declare the arrays and structures that will contain the simulation data:
	voxel_struct voxel_data;	// Define the geometric constants of the voxel file
	detector_struct detector_data[MAX_NUM_PROJECTIONS]; // Define an x ray detector (for each projection)
	source_struct source_data[MAX_NUM_PROJECTIONS]; // Define the particles source (for each projection)
	source_energy_struct source_energy_data; // Define the source energy spectrum
	linear_interp mfp_table_data; // Constant data for the linear interpolation
	compton_struct compton_table; // Structure containing Compton sampling data (to be copied to CONSTANT memory)
	rayleigh_struct rayleigh_table; // Structure containing Rayleigh sampling data (to be copied to CONSTANT memory)
    // -- Declare the pointers to the device global memory, when using the GPU:
	float2 *voxel_mat_dens_device, *mfp_Woodcock_table_device;
	float3 *mfp_table_a_device, *mfp_table_b_device;
	unsigned long long *image_device;
	rayleigh_struct *rayleigh_table_device;
	compton_struct *compton_table_device;
	ulonglong2 *voxels_Edep_device;
	detector_struct *detector_data_device;
	source_struct *source_data_device;
	ulonglong2 *materials_dose_device; // !!tally_materials_dose!!

	float2 *voxel_mat_dens; // Poiter where voxels array will be allocated
	unsigned int voxel_mat_dens_bytes; // Size (in bytes) of the voxels array (using unsigned int to allocate up to 4.2GBytes)
	float density_max[MAX_MATERIALS];
	float density_nominal[MAX_MATERIALS];
	unsigned long long int *image;// Poiter where image array will be allocated
	int image_bytes;// Size of the image array
	int mfp_table_bytes, mfp_Woodcock_table_bytes;// Size of the table arrays
	float2 *mfp_Woodcock_table;// Linear interpolation data for the Woodcock mean free path [cm]
	float3 *mfp_table_a, *mfp_table_b; // Linear interpolation data for 3 different interactions:(1) inverse total mean free path (divided by density, cm^2/g) (2) inverse Compton mean free path (divided by density, cm^2/g) (3) inverse Rayleigh mean free path (divided by density, cm^2/g)
	short dose_ROI_x_min, dose_ROI_x_max, dose_ROI_y_min, dose_ROI_y_max, dose_ROI_z_min, dose_ROI_z_max; // Coordinates of the dose region of interest (ROI)
	ulonglong2 *voxels_Edep; // Poiter where the voxel energy deposition array will be allocated
	int voxels_Edep_bytes; // Size of the voxel Edep array

	ulonglong2 materials_dose[MAX_MATERIALS]; // Array for tally_materials_dose.  

    unsigned long long int total_histories;
	int histories_per_thread, seed_input, num_threads_per_block, gpu_id, num_projections;
	int flag_material_dose;
	double D_angle, angularROI_0, angularROI_1, initial_angle, SRotAxisD, vertical_translation_per_projection;
	char file_name_voxels[250], file_name_materials[MAX_MATERIALS][250], file_name_output[250], file_dose_output[250], file_name_espc[250];
	int output_proj_type;
    float mean_energy_spectrum;
}; // class MCGPU

// helper to use singleton
#define MCGPU_MATLAB MCGPU::get_instance_ptr()

#endif