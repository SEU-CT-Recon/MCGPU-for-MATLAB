#ifndef MCGPU_COMMON_DEFINES_H_
#define MCGPU_COMMON_DEFINES_H_

//! MPI macro: mark commands to be executed only by the master thread (myID==0).
#define MASTER_THREAD if(0==myID)

//! Maximum number of projections allowed in the CT simulation (not limited by the constant memory because stored in global and shared memory):
#define  MAX_NUM_PROJECTIONS  720

//! Constants values for the Compton and Rayleigh models:
#define  MAX_MATERIALS      20
#define  MAX_SHELLS         30
#define  NP_RAYLEIGH       128
// #define  MAX_ENERGYBINS_RAYLEIGH  25005
#define  MAX_ENERGYBINS_RAYLEIGH  50000

//! Maximum number of energy bins in the input x-ray energy spectrum.
#define  MAX_ENERGY_BINS   200


#define  PI      3.14159265358979323846
#define  RAD2DEG 180.0/PI
#define  DEG2RAD PI/180.0

// Other constants used in the code:
//! Value to scale the deposited energy in the pixels so that it can be stored as a long long integer
//! instead of a double precision float. The integer values have to be used in order to use the
//! atomicAdd function in CUDA.
#define SCALE_eV        100.0f

//! Offset value to make sure the particles are completely inside, or outside, the voxel bounding box.
//! For example, to start a particle track the source may have to translate a particle from the focal spot to the plane Y=0, but 
//! in reality the particle will be moved to Y=EPS_SOURCE to make sure it is unmistakenly located inside a voxel.
//! If "EPS_SOURCE" is too small an artifact with weird concentric circular shadows may appear at the center of the simulated image.
#define EPS_SOURCE      0.000015f      

#define NEG_EPS_SOURCE -EPS_SOURCE
#define INF             500000.0f
#define INF_minus1      499999.0f
#define NEG_INF        -500000.0f

//!  Upper limit of the number of random values sampled in a single track.
#define LEAP_DISTANCE 256
//!  Multipliers and moduli for the two MLCG in RANECU.
#define a1_RANECU 40014
#define m1_RANECU 2147483563
#define a2_RANECU 40692
#define m2_RANECU 2147483399

//! Preprocessor macro to calculate maximum and minimum values:
#define max_value( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min_value( a, b ) ( ((a) < (b)) ? (a) : (b) )

#define CHECK_AND_DELETE(ptr) do {  \
    if (nullptr != (ptr)) {         \
        free(ptr);                  \
        ptr = nullptr; }            \
    } while(0)

#define CHECK_AND_DELETE_OBJ(ptr) do {      \
    if (nullptr != (ptr)) {                 \
        delete ptr;                         \
        ptr = nullptr; }                    \
    } while(0)

#define CHECK_AND_FREE(ptr) do {    \
    if (nullptr != (ptr)) {         \
        cudaFree(ptr);              \
        ptr = nullptr; }            \
    } while(0)

// Include CUDA functions:

// CUDA runtime
#include <cuda_runtime.h>
#include <mex.h>
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

template <typename T>
bool check0(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        mexPrintf("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        return false;
    }
    return true;
}

#define SAFE_CALL(val) do {                         \
    if (!check0((val), #val, __FILE__, __LINE__))   \
    return false;                                   \
} while(0)

#include <vector_types.h>

// MC-GPU structure declarations:

//! Structure storing the data defining the source model (except for the energy spectrum).
//! When a CT is simulated, multiple sources will be stored in an array (one source instance per projection angle).
struct
__align__(16)
source_struct       // Define a cone beam x-ray source.
{
  float3 position,
         direction;
  float rot_fan[9],      // Rotation (from axis (0,1,0)) defined by the source direction (needed by the fan beam source; its inverse is used by the detector).
        cos_theta_low,   // Angles for the fan beam model (pyramidal source).
        phi_low,
        D_cos_theta,
        D_phi,
        max_height_at_y1cm;
};


//! Structure storing the source energy spectrum data to be sampled using the Walker aliasing algorithm.
struct
__align__(16)
source_energy_struct       // Define a cone beam x-ray source.
{  
  int num_bins_espc;                     // Number of bins in the input energy spectrum
  float espc[MAX_ENERGY_BINS],           // Energy values of the input x-ray energy spectrum
        espc_cutoff[MAX_ENERGY_BINS];    // Cutoffs for the Walker aliasing sampling of the spectrum
  short int espc_alias[MAX_ENERGY_BINS]; // Aliases for the Walker aliasing sampling algorithm (stored as short to save memory)
};


//! Structure storing the data defining the x-ray detector. 
//! For a CT, the struct stores for each angle the detector location and the rotations to
//! transport the detector to a plane perpendicular to +Y.
//! To simulate multiple projections, an array of MAX_NUM_PROJECTIONS of instances of this struct have to be defined!  !!DeBuG!! shared
struct
__align__(16)
detector_struct         // Define a 2D detector plane, located in front of the defined source (centered at the focal spot and perpendicular to the initial direction).
{                       // The radiograohic image will be stored in the global variable "unsigned long long int *image".
  float sdd;                                // Store the source-detector distance
  float3 corner_min_rotated_to_Y,
         center;
  float rot_inv[9],    // Rotation to transport a particle on the detector plane to a frame where the detector is perpendicular to +Y.
        width_X,
        height_Z,
        inv_pixel_size_X,
        inv_pixel_size_Z;
  int2 num_pixels;
  int total_num_pixels,
      rotation_flag;    // Flag >0 if the initial source direction is not (0,1,0); ==0 otherwise (ie, detector perpendicular to +Y axis and rotation not required).
};


//! Structure defining a voxelized box with the back-lower corner at the coordinate origin.
struct
__align__(16)
voxel_struct        // Define a voxelized box with the back-lower corner at the coordinate origin.
{                                  // Voxel material and densities are stored in a local variable.
  int3 num_voxels;
  float3 inv_voxel_size,
         size_bbox;
};


//! Structure with the basic data required by the linear interpolation of the mean free paths: number of values and energy grid.
struct
__align__(16)
linear_interp       // Constant data for linear interpolation of mean free paths
{                                        // The parameters 'a' and 'b' are stored in local variables float4 *a, *b;
  int num_values;      // -->  Number of iterpolation points (eg, 2^12 = 4096).
  float e0,            // --> Minimum energy
        ide;           // --> Inverse of the energy bin width
};


//! Structure storing the data of the Compton interaction sampling model (equivalent to PENELOPE's common block /CGCO/).
struct
__align__(16)
compton_struct      // Data from PENELOPE's common block CGCO: Compton interaction data
{
  float fco[MAX_MATERIALS*MAX_SHELLS],
        uico[MAX_MATERIALS*MAX_SHELLS],
        fj0[MAX_MATERIALS*MAX_SHELLS];
  int noscco[MAX_MATERIALS];
};

//! Structure storing the data of the Rayleigh interaction sampling model (equivalent to PENELOPE's common block /CGRA/).
struct
__align__(16)
rayleigh_struct
{
  float xco[NP_RAYLEIGH*MAX_MATERIALS],
        pco[NP_RAYLEIGH*MAX_MATERIALS],
        aco[NP_RAYLEIGH*MAX_MATERIALS],
        bco[NP_RAYLEIGH*MAX_MATERIALS],
        pmax[MAX_ENERGYBINS_RAYLEIGH*MAX_MATERIALS];
  unsigned char itlco[NP_RAYLEIGH*MAX_MATERIALS],
                ituco[NP_RAYLEIGH*MAX_MATERIALS];
};

#endif