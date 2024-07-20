/*
 * @Author: Tianling Lyu
 * @Date: 2023-12-28 17:11:03
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2023-12-30 23:21:55
 * @FilePath: \firstRECON2_matlab\MATLAB\Scatter\MCGPU_mex.cpp
 */

#include <mex.h>

#include "MATLAB/Scatter/MCGPU.h"

void mexInitMCGPU(int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[])
{
    // check parameter
    if (nrhs < 2) {
        mexErrMsgTxt("Usage: suc = MCGPU_mex('init', param); \n \
            \tparam: MCGPU parameter struct");
    }
    // get parameter pointer
    mxArray* param = prhs[1];
    if (mxSTRUCT_CLASS != mxGetClassID(param)) {
        mexErrMsgTxt("Usage: suc = MCGPU_mex('init', param); \n \
            \tparam: MATLAB struct type!");
        return;
    }
    // set output
    mwSize size[3];
    size[0] = 1;
    size[1] = 1;
    size[2] = 1;
    plhs[0] = mxCreateNumericArray(1, size, mxINT32_CLASS, mxREAL);
    int* out_ptr = reinterpret_cast<int*>(mxGetData(plhs[0]));
    *out_ptr = 0;
    *out_ptr = MCGPU_MATLAB->initialize(param);
    if (!(*out_ptr)) {
        mexErrMsgTxt("Failed in initializing simulation!");
    }
    return;
}

void mexRunMCGPU(int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[])
{
    // check parameter
    if (nrhs < 2) {
        mexErrMsgTxt("Usage: suc = MCGPU_mex('run', voxel_data);");
    }
    // get parameter pointer
    mxArray* param = prhs[1];
    if (mxSTRUCT_CLASS != mxGetClassID(param)) {
        mexErrMsgTxt("Usage: suc = MCGPU_mex('init', param); \n \
            \tvoxel_data: data struct!");
        return;
    }
    // set output
    mwSize size[3];
    size[0] = MCGPU_MATLAB->getProjWidth();
    size[1] = MCGPU_MATLAB->getProjHeight();
    size[2] = MCGPU_MATLAB->getProjNum();
    plhs[0] = mxCreateNumericArray(3, size, mxSINGLE_CLASS, mxREAL);
    float* noScatter = reinterpret_cast<float*>(mxGetData(plhs[0]));
    plhs[1] = mxCreateNumericArray(3, size, mxSINGLE_CLASS, mxREAL);
    float* compton = reinterpret_cast<float*>(mxGetData(plhs[1]));
    plhs[2] = mxCreateNumericArray(3, size, mxSINGLE_CLASS, mxREAL);
    float* rayleigh = reinterpret_cast<float*>(mxGetData(plhs[2]));
    plhs[3] = mxCreateNumericArray(3, size, mxSINGLE_CLASS, mxREAL);
    float* multiscatter = reinterpret_cast<float*>(mxGetData(plhs[3]));
    if (!MCGPU_MATLAB->run(param, noScatter, compton, rayleigh, multiscatter))
        mexErrMsgTxt("Failed in running simulation!");
    return;
}

void mexClearMCGPU(int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[])
{
    // set output
    mwSize size[3];
    size[0] = 1;
    size[1] = 1;
    size[2] = 1;
    plhs[0] = mxCreateNumericArray(1, size, mxINT32_CLASS, mxREAL);
    int* out_ptr = reinterpret_cast<int*>(mxGetData(plhs[0]));
    *out_ptr = 0;
    *out_ptr = MCGPU_MATLAB->clear();
    if (!(*out_ptr)) {
        mexErrMsgTxt("Failed in clearing simulation!");
    }
    return;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[])
{
    if (nrhs < 1) {
        mexErrMsgTxt("Usage: MCGPU_mex(operation, ...); \n\
        \t operation: 'init', 'run' or 'clear'.");
    }
    std::string op = mxArrayToString(prhs[0]);
    if (op == "init") {
        mexInitMCGPU(nlhs, plhs, nrhs, prhs);
    } else if (op == "run") {
        mexRunMCGPU(nlhs, plhs, nrhs, prhs);
    } else if (op == "clear") {
        mexClearMCGPU(nlhs, plhs, nrhs, prhs);
    } else {
        mexErrMsgTxt("Unrecognized operator operation!");
    }
    return;
}