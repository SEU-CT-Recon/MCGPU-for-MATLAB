/*
 * @Author: Tianling Lyu
 * @Date: 2022-08-26 09:57:47
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2022-09-06 16:20:34
 * @FilePath: \mcgpu_python\src\constants.h
 */
#ifndef MCGPU_CONSTANTS_H_
#define MCGPU_CONSTANTS_H_

#include "MATLAB/Scatter/common_defines.h"

//// *** DEVICE CONSTANT MEMORY DECLARATION (~global variables in the GPU) *** ////

// -- Constant memory (defined as global variables):

//! Global variable to be stored in the GPU constant memory defining the coordinates of the dose deposition region of interest.
__constant__
short int dose_ROI_x_min_CONST, dose_ROI_x_max_CONST, dose_ROI_y_min_CONST, dose_ROI_y_max_CONST, dose_ROI_z_min_CONST, dose_ROI_z_max_CONST;

//! Global variable to be stored in the GPU constant memory defining the size of the voxel phantom.
__constant__
struct voxel_struct    voxel_data_CONST;      // Define the geometric constants

//! Global variable to be stored in the GPU constant memory defining the x-ray source.
__constant__
struct source_struct   source_data_CONST;     // Define a particles source.

//! Global variable to be stored in the GPU constant memory defining the x-ray detector.
__constant__
struct detector_struct detector_data_CONST;   // Define a detector layer perpendicular to the y axis

//! Global variable to be stored in the GPU constant memory defining the linear interpolation data.
__constant__
struct linear_interp   mfp_table_data_CONST;  // Define size of interpolation arrays

//! Global variable to be stored in the GPU constant memory defining the source energy spectrum.
__constant__
struct source_energy_struct source_energy_data_CONST;

#endif