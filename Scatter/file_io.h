/*
 * @Author: Tianling Lyu
 * @Date: 2022-08-24 11:03:00
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2022-09-06 17:25:10
 * @FilePath: \mcgpu_python\src\file_io.h
 */
#ifndef MCGPU_FILEIO_H_
#define MCGPU_FILEIO_H_
#include "common_defines.h"
#include <zlib.h>

////////////////////////////////////////////////////////////////////////////////
//! Read a line of text and trim initial blancks and trailing comments (#).
//!
//!       @param[in] num   Characters to read
//!       @param[in] file_ptr   Pointer to the input file stream
//!       @param[out] trimmed_line   Trimmed line from input file, skipping empty lines and comments
////////////////////////////////////////////////////////////////////////////////
char *fgets_trimmed(char *trimmed_line, int num, FILE *file_ptr);

////////////////////////////////////////////////////////////////////////////////
//! Read the voxel data and allocate the material and density matrix.
//! Also find and report the maximum density defined in the geometry.
//!
// -- Sample voxel geometry file:
//
//   #  (comment lines...)
//   #
//   #   Voxel order: X runs first, then Y, then Z.
//   #
//   [SECTION VOXELS HEADER v.2008-04-13]
//   411  190  113      No. OF VOXELS IN X,Y,Z
//   5.000e-02  5.000e-02  5.000e-02    VOXEL SIZE (cm) ALONG X,Y,Z
//   1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED
//   2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED
//   1                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)
//   [END OF VXH SECTION]
//   1 0.00120479
//   1 0.00120479
//   ...
//
//!       @param[in] file_name_voxels  Name of the voxelized geometry file.
//!       @param[out] density_max  Array with the maximum density for each material in the voxels.
//!       @param[out] voxel_data   Pointer to a structure containing the voxel number and size.
//!       @param[out] voxel_mat_dens_ptr   Pointer to the vector with the voxel materials and densities.
//!       @param[in] dose_ROI_x/y/z_max   Size of the dose ROI: can not be larger than the total number of voxels in the geometry.
////////////////////////////////////////////////////////////////////////////////
bool load_voxels(char *file_name_voxels, float *density_max, 
    struct voxel_struct *voxel_data, float2 **voxel_mat_dens_ptr, 
    unsigned int *voxel_mat_dens_bytes, short int *dose_ROI_x_max, 
    short int *dose_ROI_y_max, short int *dose_ROI_z_max);

////////////////////////////////////////////////////////////////////////////////
//! Read the material input files and set the mean free paths and the "linear_interp" structures.
//! Find the material nominal density. Set the Woodcock trick data.
//
// -- Sample material data file (data obtained from the PENELOPE 2006 database and models):
//
//    [MATERIAL NAME]
//     Water
//    [NOMINAL DENSITY (g/cm^3)]
//     1.000
//    [NUMBER OF DATA VALUES]
//     4096
//    [MEAN FREE PATHS :: Energy (eV) || Rayleigh | Compton | Photoelectric | Pair-production | TOTAL (cm)]
//     1.00000E+03  7.27451E-01  9.43363E+01  2.45451E-04  1.00000E+35  2.45367E-04
//     5.00000E+03  1.80004E+00  8.35996E+00  2.38881E-02  1.00000E+35  2.35089E-02
//     1.00000E+04  4.34941E+00  6.26746E+00  2.02568E-01  1.00000E+35  1.87755E-01
//     ...
//     #[RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database)]
//     ...
//     #[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
//     ...
//
//!       @param[in] file_name_materials    Array with the names of the material files.
//!       @param[in] density_max   maximum density in the geometry (needed to set Woodcock trick)
//!       @param[out] density_nominal   Array with the nominal density of the materials read
//!       @param[out] mfp_table_data   Constant values for the linear interpolation
//!       @param[out] mfp_table_a_ptr   First element for the linear interpolation.
//!       @param[out] mfp_table_b_ptr   Second element for the linear interpolation.
////////////////////////////////////////////////////////////////////////////////
bool load_material(char file_name_materials[MAX_MATERIALS][250], float *density_max, float *density_nominal, struct linear_interp *mfp_table_data, float2 **mfp_Woodcock_table_ptr, int *mfp_Woodcock_table_bytes, float3 **mfp_table_a_ptr, float3 **mfp_table_b_ptr, int *mfp_table_bytes, struct rayleigh_struct *rayleigh_table_ptr, struct compton_struct *compton_table_ptr);

////////////////////////////////////////////////////////////////////////////////
//! Report the tallied image in ASCII and binary form (32-bit floats).
//! Separate images for primary and scatter radiation are generated.
//!
//!
//!       @param[in] file_name_output   File where tallied image is reported
//!       @param[in] detector_data   Detector description read from the input file (pointer to detector_struct)
//!       @param[in] image  Tallied image (in meV per pixel)
//!       @param[in] time_elapsed   Time elapsed during the main loop execution (in seconds)
//!       @param[in] total_histories   Total number of x-rays simulated
////////////////////////////////////////////////////////////////////////////////
int report_image(char *file_name_output, int r_output_proj_type, struct detector_struct *detector_data, struct source_struct *source_data, float mean_energy_spectrum, unsigned long long int *image, double time_elapsed, unsigned long long int total_histories, int current_projection, int num_projections, double D_angle, double initial_angle);

///////////////////////////////////////////////////////////////////////////////
//! Report the total tallied 3D voxel dose deposition for all projections.
//! The voxel doses in the input ROI and their respective uncertainties are reported
//! in binary form (32-bit floats) in two separate .raw files.
//! The dose in a single plane at the level of the focal spot is also reported in
//! ASCII format for simple visualization with GNUPLOT.
//! The total dose deposited in each different material is reported to the standard output.
//! The material dose is calculated adding the energy deposited in the individual voxels
//! within the dose ROI, and dividing by the total mass of the material in the ROI.
//!
//!       @param[in] file_dose_output   File where tallied image is reported
//!       @param[in] detector_data   Detector description read from the input file (pointer to detector_struct)
//!       @param[in] image  Tallied image (in meV per pixel)
//!       @param[in] time_elapsed   Time elapsed during the main loop execution (in seconds)
//!       @param[in] total_histories   Total number of x-rays simulated
//!       @param[in] source_data   Data required to compute the voxel plane to report in ASCII format: Z at the level of the source, 1st projection
////////////////////////////////////////////////////////////////////////////////
int report_voxels_dose(char *file_dose_output, int num_projections, struct voxel_struct *voxel_data, float2 *voxel_mat_dens, ulonglong2 *voxels_Edep, double time_total_MC_init_report, unsigned long long int total_histories, short int dose_ROI_x_min, short int dose_ROI_x_max, short int dose_ROI_y_min, short int dose_ROI_y_max, short int dose_ROI_z_min, short int dose_ROI_z_max, struct source_struct *source_data);

///////////////////////////////////////////////////////////////////////////////
//! Report the tallied dose to each material number, accounting for different
//! densities in different regions with the same material number.
//!
//!       @param[in] num_projections   Number of projections simulated
//!       @param[in] total_histories   Total number of x-rays simulated per projection
//!       @param[out] density_nominal   Array with the nominal densities of materials given in the input file; -1 for materials not defined. Used to report only defined materials.
//!       @param[in] materials_dose   Tallied dose and dose^2 arrays
////////////////////////////////////////////////////////////////////////////////
int report_materials_dose(int num_projections, unsigned long long int total_histories, float *density_nominal, ulonglong2 *materials_dose, double *mass_materials);

#endif