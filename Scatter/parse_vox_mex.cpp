/*
 * @Author: Tianling Lyu
 * @Date: 2023-12-30 21:01:57
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2023-12-30 21:58:57
 * @FilePath: \firstRECON2_matlab\MATLAB\Scatter\parse_vox_mex.cpp
 */

#include <mex.h>
#include <fstream>

#define MAX_MATERIALS 20

struct float3
{
    double x, y, z;
};

bool load_voxels(char *filename, float* material, float* density, float* density_max, float3 num_voxels, float3 voxel_size)
{
    char new_line[250];
    char *new_line_ptr = NULL;

    FILE *file_ptr;
    fopen_s(&file_ptr, filename, "rb");

    if (file_ptr == NULL) {
        mexErrMsgTxt("!! fopen ERROR load_voxels!! File does not exist!!\n");
        return false;
    }
    
    do {
        new_line_ptr = fgets(new_line, 250, file_ptr);
        if (new_line_ptr == NULL)
        {
            mexErrMsgTxt("!!Reading ERROR load_voxels!! File is not readable or does not contain the string \'[END OF VXH SECTION]\'!!\n");
            return false;
        }
    } while (strstr(new_line, "[END OF VXH SECTION") == NULL); // headers

    // -- Read the voxel densities:
    int i, j, k, read_lines = 0, dummy_material, read_items = -99;
    float dummy_density;
    float *mat_ptr = material;
    float* den_ptr = density;

    for (k = 0; k < MAX_MATERIALS; k++)
        density_max[k] = -999.0f; // Init array with an impossible low density value

    for (k = 0; k < (num_voxels.z); k++) {
        for (j = 0; j < (num_voxels.y); j++) {
            for (i = 0; i < (num_voxels.x); i++) {
                do {
                    new_line_ptr = fgets(new_line, 250, file_ptr);                                                    //  !!zlib!!
                } while (('\n' == new_line[0]) || ('\n' == new_line[1]) || ('#' == new_line[0]) || ('#' == new_line[1])); // Skip empty lines and comments.
                read_items = sscanf(new_line, "%d %f", &dummy_material, &dummy_density);                                  // Read the next 2 numbers

                if (read_items != 2)
                    mexPrintf("\n   !!WARNING load_voxels!! Expecting to read 2 items (material and density). read_items=%d, read_lines=%d \n", read_items, read_lines);

                if (dummy_material > MAX_MATERIALS) {
                    mexPrintf("\n\n   !!ERROR load_voxels!! Voxel material number too high!! #mat=%d, MAX_MATERIALS=%d, voxel number=%d\n\n", dummy_material, MAX_MATERIALS, read_lines + 1);
                    return false;
                }
                if (dummy_material < 1) {
                    mexPrintf("\n\n   !!ERROR load_voxels!! Voxel material number can not be zero or negative!! #mat=%d, voxel number=%dd\n\n", dummy_material, read_lines + 1);
                    return false;
                }

                if (dummy_density < 1.0e-9f) {
                    mexPrintf("\n\n   !!ERROR load_voxels!! Voxel density can not be 0 or negative: #mat=%d, density=%f, voxel number=%d\n\n", dummy_material, dummy_density, read_lines + 1);
                    return false;
                }

                if (dummy_density > density_max[dummy_material - 1])
                    density_max[dummy_material - 1] = dummy_density; // Store maximum density for each material

                *mat_ptr = (float)(dummy_material) + 0.0001f; // Assign material value as float (the integer value will be recovered by truncation)
                *den_ptr = dummy_density;                     // Assign density value

                mat_ptr++;
                den_ptr++;
                read_lines++;
            }
        }
    }
    mexPrintf("       Total number of voxels read: %d\n", read_lines);
    fclose(file_ptr); // Close input file    !!zlib!!
    return true;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[])
{
    if (nrhs < 3) {
        mexErrMsgTxt("Usage: [material, density, density_max]=parse_vox_mex(filename, num_voxels, voxel_size);");
    }
    if (nlhs < 3) {
        mexErrMsgTxt("Usage: [material, density, density_max]=parse_vox_mex(filename, num_voxels, voxel_size);");
    }
    // parse inputs
    char filename[256];
    mxGetString(prhs[0], filename, 256);
    if (!mxIsDouble(prhs[1])) {
        mexErrMsgTxt("Usage: [material, density, density_max]=parse_vox_mex(filename, num_voxels, voxel_size);\n \tnum_voxels: double array with 3 elements.");
    }
    float3* num_voxels = reinterpret_cast<float3*>(mxGetData(prhs[1]));
    if (!mxIsDouble(prhs[2])) {
        mexErrMsgTxt("Usage: [material, density, density_max]=parse_vox_mex(filename, num_voxels, voxel_size);\n \tvoxel_size: double array with 3 elements.");
    }
    float3* voxel_size = reinterpret_cast<float3*>(mxGetData(prhs[2]));
    // construct outputs
    mwSize size[3];
    size[0] = num_voxels->x;
    size[1] = num_voxels->y;
    size[2] = num_voxels->z;
    plhs[0] = mxCreateNumericArray(3, size, mxSINGLE_CLASS, mxREAL);
    float* material = reinterpret_cast<float*>(mxGetData(plhs[0]));
    plhs[1] = mxCreateNumericArray(3, size, mxSINGLE_CLASS, mxREAL);
    float* density = reinterpret_cast<float*>(mxGetData(plhs[1]));
    size[0] = MAX_MATERIALS;
    size[1] = 1;
    size[2] = 1;
    plhs[2] = mxCreateNumericArray(1, size, mxSINGLE_CLASS, mxREAL);
    float* density_max = reinterpret_cast<float*>(mxGetData(plhs[2]));
    if (!load_voxels(filename, material, density, density_max, *num_voxels, *voxel_size))
    {
        mexErrMsgTxt("Failed in load_voxels!\n");
    }
    return;
}