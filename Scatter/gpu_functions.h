/*
 * @Author: Tianling Lyu
 * @Date: 2022-08-26 15:51:24
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2023-12-28 14:57:37
 * @FilePath: \firstRECON2_matlab\MATLAB\Scatter\gpu_functions.h
 */
#ifndef MCGPU_GPU_FUNCTIONS_H_
#define MCGPU_GPU_FUNCTIONS_H_

#include "common_defines.h"

int guestimate_GPU_performance(int gpu_id);

bool init_CUDA_device( int* gpu_id, int myID, int numprocs,
    /*Variables to GPU constant memory:*/ struct voxel_struct* voxel_data, struct source_struct* source_data, struct source_energy_struct* source_energy_data, struct detector_struct* detector_data, struct linear_interp* mfp_table_data,
    /*Variables to GPU global memory:*/ float2* voxel_mat_dens, float2** voxel_mat_dens_device, unsigned int voxel_mat_dens_bytes,
    unsigned long long int* image, unsigned long long int** image_device, int image_bytes,
    float2* mfp_Woodcock_table, float2** mfp_Woodcock_table_device, int mfp_Woodcock_table_bytes,
    float3* mfp_table_a, float3* mfp_table_b, float3** mfp_table_a_device, float3** mfp_table_b_device, int mfp_table_bytes,
    struct rayleigh_struct* rayleigh_table, struct rayleigh_struct** rayleigh_table_device,
    struct compton_struct* compton_table, struct compton_struct** compton_table_device,
    struct detector_struct** detector_data_device, struct source_struct** source_data_device,
    ulonglong2* voxels_Edep, ulonglong2** voxels_Edep_device, int voxels_Edep_bytes, short int* dose_ROI_x_min, short int* dose_ROI_x_max, short int* dose_ROI_y_min, short int* dose_ROI_y_max, short int* dose_ROI_z_min, short int* dose_ROI_z_max,
    ulonglong2* materials_dose, ulonglong2** materials_dose_device, int flag_material_dose, int num_projections);

//// *** GLOBAL FUNCTIONS *** ////
bool init_image_array_GPU(unsigned long long int* image, int pixels_per_image);
bool track_particles(dim3 block, dim3 thread, int histories_per_thread, int num_p, int seed_input, unsigned long long int* image, ulonglong2* voxels_Edep, float2* voxel_mat_dens, float2* mfp_Woodcock_table, float3* mfp_table_a, float3* mfp_table_b, struct rayleigh_struct* rayleigh_table, struct compton_struct* compton_table, struct detector_struct* detector_data_array, struct source_struct* source_data_array, ulonglong2* materials_dose);


//// *** DEVICE FUNCTIONS *** ////
__device__
inline void source(float3* position, float3* direction, float* energy, int2* seed, int* absvox, struct source_struct* source_data_SHARED, struct detector_struct* detector_data_SHARED);
__device__
inline void move_to_bbox(float3* position, float3* direction, float3 size_bbox, int* intersection_flag);
__device__
inline void tally_image(float* energy, float3* position, float3* direction, signed char* scatter_state, unsigned long long int* image, source_struct* source_data_SHARED, detector_struct* detector_data_SHARED);
__device__
inline void init_PRNG(int history_batch, int histories_per_thread, int seed_input, int2* seed);
__host__ __device__
int abMODm(int m, int a, int s);
__device__
inline float ranecu(int2* seed);
__device__
inline double ranecu_double(int2* seed);
__device__
inline int locate_voxel(float3* position, short3* voxel_coord);
__device__
inline void rotate_double(float3* direction, double cos_theta, double phi);
__device__
inline void GRAa(float *energy, double *costh_Rayleigh, int *mat, float *pmax_current, int2 *seed, struct rayleigh_struct* cgra);
__device__
inline void GCOa(float *energy, double *costh_Compton, int *mat, int2 *seed, struct compton_struct* cgco_SHARED);

__device__
inline void tally_voxel_energy_deposition(float* Edep, short3* voxel_coord, ulonglong2* dose);

__device__
inline 
void tally_materials_dose(float* Edep, int* material, ulonglong2* materials_dose);

#endif