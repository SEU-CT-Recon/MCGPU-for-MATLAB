/*
 * @Author: Tianling Lyu
 * @Date: 2022-09-06 10:03:40
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2023-12-31 11:15:57
 * @FilePath: \firstRECON2_matlab\MATLAB\Scatter\MCGPU.cpp
 */
#include "MATLAB/Scatter/MCGPU.h"

#include "MATLAB/Scatter/file_io.h"
#include "MATLAB/Scatter/gpu_functions.h"

extern bool ioFlush(void);

DEFINE_SINGLETON(MCGPU);

MCGPU::MCGPU()
    : voxel_mat_dens(nullptr), voxel_mat_dens_bytes(0), image(nullptr), 
    mfp_table_bytes(-1), mfp_Woodcock_table_bytes(-1), mfp_table_a(nullptr), 
    mfp_table_b(nullptr), mfp_Woodcock_table(nullptr), voxels_Edep(nullptr), 
    voxels_Edep_bytes(0), D_angle(-1.0), angularROI_0(0.0), angularROI_1(0.0), 
    initial_angle(0.0), SRotAxisD(-1.0), vertical_translation_per_projection(0.0), 
    output_proj_type(0), mean_energy_spectrum(0.0)
{
    // -- Declare the pointers to the device global memory, when using the GPU:
	voxel_mat_dens_device = NULL;
	mfp_Woodcock_table_device = NULL;
	mfp_table_a_device = NULL;
	mfp_table_b_device = NULL;
	image_device = NULL;
	rayleigh_table_device = NULL;
	compton_table_device = NULL;
	voxels_Edep_device = NULL;
	detector_data_device = NULL;
	source_data_device = NULL;
	materials_dose_device = NULL; // !!tally_materials_dose!!
    for (size_t kk = 0; kk < MAX_MATERIALS; kk++)
	{
		materials_dose[kk].x = 0; // Initializing data !!tally_materials_dose!!
		materials_dose[kk].y = 0;
		density_nominal[kk] = -1.0f;
	}
}

MCGPU::~MCGPU()
{
    clear();
}

bool MCGPU::initialize(mxArray* params)
{
	// cleanup params and pointers
	CHECK_AND_DELETE(voxel_mat_dens);
	voxel_mat_dens_bytes = 0;
	CHECK_AND_DELETE(image);
	mfp_table_bytes = -1;
	mfp_Woodcock_table_bytes = -1;
	CHECK_AND_DELETE(mfp_table_a);
    CHECK_AND_DELETE(mfp_table_b);
	CHECK_AND_DELETE(mfp_Woodcock_table);
	CHECK_AND_DELETE(voxels_Edep);
    voxels_Edep_bytes = 0;
	D_angle = -1.0;
	angularROI_0 = 0.0;
	angularROI_1 = 0.0; 
    initial_angle = 0.0;
	SRotAxisD = -1.0;
	vertical_translation_per_projection = 0.0; 
    output_proj_type = 0;
	mean_energy_spectrum = 0.0;
	// -- cleanup the pointers to the device global memory, when using the GPU:
	CHECK_AND_FREE(voxel_mat_dens_device);
	CHECK_AND_FREE(mfp_Woodcock_table_device);
	CHECK_AND_FREE(mfp_table_a_device);
	CHECK_AND_FREE(mfp_table_b_device);
	CHECK_AND_FREE(image_device);
	CHECK_AND_FREE(rayleigh_table_device);
	CHECK_AND_FREE(compton_table_device);
	CHECK_AND_FREE(voxels_Edep_device);
	CHECK_AND_FREE(detector_data_device);
	CHECK_AND_FREE(source_data_device);
	CHECK_AND_FREE(materials_dose_device);
    for (size_t kk = 0; kk < MAX_MATERIALS; kk++)
	{
		materials_dose[kk].x = 0; // Initializing data !!tally_materials_dose!!
		materials_dose[kk].y = 0;
		density_nominal[kk] = -1.0f;
	}
	_print_init();

    // *** Parse input MATLAB struct:
	if (!parse_param_struct(params)) {
		mexPrintf("Failed to parse parameters!\n");
		return false;
	}

    // *** Read the energy spectrum and initialize its sampling with the Walker aliasing method:
	/*mean_energy_spectrum = 0.0f;
	if (!init_energy_spectrum(file_name_espc, &source_energy_data, &mean_energy_spectrum))
		return false;*/

    _print_read();
	ioFlush();

    	// *** Set the detectors and sources for the CT trajectory (if needed, ie, for more than one projection):
	if (num_projections != 1)
	{
		if (!set_CT_trajectory(num_projections, D_angle, angularROI_0, 
			angularROI_1, SRotAxisD, source_data, detector_data, 
			vertical_translation_per_projection))
			return false;
	}
	ioFlush();
	
	return true;
}

bool MCGPU::run(mxArray* param, float* noScatter, float* compton, float* rayleigh, 
	float* multiscatter)
{
    // -- Start time counter:
	time_t current_time = time(NULL);					   // Get current time (in seconds)
	clock_t clock_start, clock_end, clock_start_beginning; // (requires standard header <time.h>)
	clock_start = clock();
    clock_start_beginning = clock_start;
    clock_t clock_kernel; // Using only cpu timers after CUDA 5.0

	double time_elapsed_MC_loop = 0.0, time_total_MC_simulation = 0.0, time_total_MC_init_report = 0.0;

	// *** Read the voxel data and allocate the density map matrix. Return the maximum density:
	if (!parse_data_struct(param))
		return false;
	_print_allocated();

	// *** Read the material mean free paths and set the interaction table in a "linear_interp" structure:
	if (!load_material(file_name_materials, density_max, density_nominal, &mfp_table_data, &mfp_Woodcock_table, &mfp_Woodcock_table_bytes, &mfp_table_a, &mfp_table_b, &mfp_table_bytes, &rayleigh_table, &compton_table))
		return false;

	// -- Check that the input material tables and the x-ray source are consistent:
	if (!_check_consistent()) return false;

	// -- Pre-compute the total mass of each material present in the voxel phantom (to be used in "report_materials_dose"):
	double voxel_volume = 1.0 / (((double)voxel_data.inv_voxel_size.x) * ((double)voxel_data.inv_voxel_size.y) * ((double)voxel_data.inv_voxel_size.z));
	double mass_materials[MAX_MATERIALS];
	for (size_t kk = 0; kk < MAX_MATERIALS; kk++)
		mass_materials[kk] = 0.0;
	for (size_t kk = 0; kk < (voxel_data.num_voxels.x * voxel_data.num_voxels.y * voxel_data.num_voxels.z); kk++) // For each voxel in the geometry
		mass_materials[((int)voxel_mat_dens[kk].x) - 1] += ((double)voxel_mat_dens[kk].y) * voxel_volume;  // Add material mass = density*volume

		// *** Initialize the GPU using the NVIDIA CUDA libraries

	// -- Sets the CUDA enabled GPU that will be used in the simulation, and allocate and copies the simulation data in the GPU global and constant memories.
	if (!init_CUDA_device(&gpu_id, 0, 1, &voxel_data, source_data, &source_energy_data, detector_data, &mfp_table_data, /*Variables GPU constant memory*/
					 voxel_mat_dens, &voxel_mat_dens_device, voxel_mat_dens_bytes, /*Variables GPU global memory*/
					 image, &image_device, image_bytes,
					 mfp_Woodcock_table, &mfp_Woodcock_table_device, mfp_Woodcock_table_bytes,
					 mfp_table_a, mfp_table_b, &mfp_table_a_device, &mfp_table_b_device, mfp_table_bytes,
					 &rayleigh_table, &rayleigh_table_device,
					 &compton_table, &compton_table_device, &detector_data_device, &source_data_device, 
					 voxels_Edep, &voxels_Edep_device, voxels_Edep_bytes, &dose_ROI_x_min, &dose_ROI_x_max, &dose_ROI_y_min, &dose_ROI_y_max, &dose_ROI_z_min, &dose_ROI_z_max,
					 materials_dose, &materials_dose_device, flag_material_dose, num_projections)
	)
		return false;

	_print_mc_start(clock_start);
	ioFlush();

	// -- A number of histories smaller than 24 hours in sec (3600*24=86400) means that the user wants to simulate for the input number of seconds in each GPU, not a fix number of histories:
	unsigned long long int total_histories_INPUT = total_histories; // Save the original input values to be re-used for multiple projections
	int seed_input_INPUT = seed_input, doing_speed_test = -1;
	int simulating_by_time = 0; // 0==false
	if (total_histories < (unsigned long long int)(7200))
		simulating_by_time = 1; // 1=true

	int num_blocks_speed_test = 0;
	unsigned long long int histories_speed_test = (unsigned long long int)0, total_histories_speed_test = (unsigned long long int)0;
	float node_speed = -1.0f, total_speed = 1.0f;
	double current_angle;

	// *** CT simulation: find the current projection angle and start Monte Carlo simulation:
	float* ns_ptr = noScatter;
	float* cp_ptr = compton;
	float* rl_ptr = rayleigh;
	float* ms_ptr = multiscatter;
	size_t pixel_num = detector_data[0].total_num_pixels;
	for (int num_p = 0; num_p < num_projections; num_p++)
	{
		// -- Check if this projection is inside the input angular region of interest (the angle can be negative, or larger than 360 in helical scans):
		current_angle = initial_angle + num_p * D_angle;

		if (!_check_angle(current_angle, num_p)) continue;

		clock_start = clock(); // Start the CPU clock

		// *** Simulate in the GPUs the input amount of time or amount of particles:
		// -- Estimate GPU speed to use a total simulation time or multiple GPUs:
		if (simulating_by_time == 0 && // Simulating a fixed number of particles, not a fixed time (so performing the speed test only once)
			node_speed > 0.0f// Speed test already performed for a previous projection in this simulation (node_speed and total_speed variables set)
            )			   // Using multiple GPUs (ie, multiple MPI threads)
		{
			// -- Simulating successive projections after the first one with a fix number of particles, with multiple MPI threads: re-use the speed test results from the first projection image:
			total_histories = (unsigned long long int)(0.5 + ((double)total_histories_INPUT) * (((double)node_speed) / total_speed));
			doing_speed_test = 0; // No speed test for this projection.
		}
		else if (simulating_by_time == 1)
		{
			// -- Simulating with a time limit OR multiple MPI threads for the first time (num_p==0): run a speed test to calculate the speed of the current GPU and distribute the number of particles to the multiple GPUs or estimate the total number of particles required to run the input amount of time:
			//    Note that this ELSE IF block will be skipped if we are using a single MPI thread and a fix number of particles.

			doing_speed_test = 1; // Remember that we are performing the speed test to make sure we add the test histories to the total before the tally reports.

			if (node_speed < 0.0f) // Speed test not performed before (first projection being simulated): set num_blocks_speed_test and histories_speed_test.
			{
				num_blocks_speed_test = guestimate_GPU_performance(gpu_id); // Guestimating a good number of blocks to estimate the speed of different generations of GPUs. Slower GPUs will simulate less particles and hopefully the fastest GPUs will not have to wait much.

				// !!DeBuG!! Error in code version 1.2 has been corrected here. Old code:   histories_speed_test = (unsigned long long int)(num_blocks_speed_test*num_threads_per_block)*(unsigned long long int)(histories_per_thread);
			}

			histories_speed_test = (unsigned long long int)(num_blocks_speed_test * num_threads_per_block) * (unsigned long long int)(histories_per_thread);

			// Re-load the input total number of histories and the random seed:
			total_histories = total_histories_INPUT;
			seed_input = seed_input_INPUT;

			dim3 blocks_speed_test(num_blocks_speed_test, 1);
			dim3 threads_speed_test(num_threads_per_block, 1);

			// -- Init the current random number generator seed to avoid overlapping sequences with other MPI threads:
			if (simulating_by_time == 1)
				// Simulating by time: set an arbitrary huge number of particles to skip.
				update_seed_PRNG((num_p), (unsigned long long int)(123456789012), &seed_input); // Set the random number seed far from any other MPI thread (myID) and away from the seeds used in the previous projections (num_p*numprocs).
			else
				// Simulating by histories
				update_seed_PRNG((num_p), total_histories, &seed_input); //  Using different random seeds for each projection
            
			mexPrintf("        ==> CUDA: Estimating the GPU speed executing %d blocks of %d threads, %d histories per thread: %lld histories in total.\n", num_blocks_speed_test, num_threads_per_block, histories_per_thread, histories_speed_test);

			clock_kernel = clock();
			// -- Launch Monte Carlo simulation kernel for the speed test:
			if (!_track_particles(blocks_speed_test, threads_speed_test, num_p))
				return false;

			float speed_test_time = float(clock() - clock_kernel) / CLOCKS_PER_SEC;
			node_speed = (float)(((double)histories_speed_test) / speed_test_time);
			mexPrintf("                  Estimated GPU speed = %lld hist / %.3f s = %.3f hist/s\n", histories_speed_test, speed_test_time, node_speed);

			// -- Init random number generator seed to avoid repeating the random numbers used in the speed test:
			update_seed_PRNG(1, histories_speed_test, &seed_input);

			if (simulating_by_time == 1)
			{
				// -- Set number of histories for each GPU when simulating by time:
				if (total_histories > speed_test_time)
					total_histories = (total_histories - speed_test_time) * node_speed; // Calculate the total number of remaining histories by "GPU speed" * "remaining time"
				else
					total_histories = 1; // Enough particles simulated already, simulate just one more history (block) and report (kernel call would fail if total_histories < or == 0).
			}
			else
			{
				total_speed = node_speed;
				// - Divide the remaining histories among the MPI threads (GPUs) according to their fraction of the total speed (rounding up).
				if (total_histories_speed_test < total_histories)
					total_histories = (unsigned long long int)(0.5 + ((double)(total_histories - total_histories_speed_test)) * ((double)(node_speed / total_speed)));
				else
					total_histories = 1; // Enough particles simulated already, simulate just one more history (block) and report (kernel call would fail if total_histories < or == 0).
			}

		} // [Done with case of simulating projections by time or first projection by number of particles]

		// *** Perform the MC simulation itself (the speed test would be skipped for a single CPU thread using a fix number of histories):
		int total_threads, total_threads_blocks;
		_check_block(total_threads, total_threads_blocks);
		ioFlush();

		// -- Setup the execution parameters (Max number threads per block: 512, Max sizes each dimension of grid: 65535x65535x1)

		dim3 blocks(total_threads_blocks, 1);
		dim3 threads(num_threads_per_block, 1);

		clock_kernel = clock();

		// *** Execute the x-ray transport kernel in the GPU ***
		if (!_track_particles(blocks, threads, num_p)) return false;

		if (1 == doing_speed_test)
			total_histories += histories_speed_test; // Speed test was done: compute the total number of histories including the particles simulated in the speed test

		// -- Move the pseudo-random number generator seed ahead to skip all the random numbers generated in the current projection by this and the other
		//    "numprocs" MPI threads. Each projection will use independent seeds! (this code runs in parallel with the asynchronous GPU kernel):
		update_seed_PRNG(1, total_histories, &seed_input); // Do not repeat seed for each projection. Note that this function only updates 1 seed, the other is not computed.

		float real_GPU_speed = float(total_histories) / (float(clock() - clock_kernel) / CLOCKS_PER_SEC); // GPU speed for all the image simulation, not just the speed test.

		// -- Copy the simulated image from the GPU memory to the CPU:
		SAFE_CALL(cudaMemcpy(image, image_device, image_bytes, cudaMemcpyDeviceToHost)); // Copy final results to host

		///////////////////////////////////////////////////////////////////////////////////////////////////
		// Get current time and calculate execution time in the MC loop:
		time_elapsed_MC_loop = ((double)(clock() - clock_start)) / CLOCKS_PER_SEC;
		time_total_MC_simulation += time_elapsed_MC_loop; // Count total time (in seconds).
														  //  mexPrintf("\n    -- MONTE CARLO LOOP finished: time tallied in MAIN program: %.3f s\n\n", time_elapsed_MC_loop);

		///////////////////////////////////////////////////////////////////////////////////////////////////

		// *** Move the images simulated in the GPU (or multiple CPU cores) to the host memory space:
		// *** Report the final results:
		char file_name_output_num_p[253];
		if (1 == num_projections)
			strcpy(file_name_output_num_p, file_name_output); // Use the input name for single projection
		else
			sprintf(file_name_output_num_p, "%s_%04d", file_name_output, num_p); // Create the output file name with the input name + projection number (4 digits, padding with 0)

		// report_image(file_name_output_num_p, output_proj_type, detector_data, source_data, mean_energy_spectrum, image, time_elapsed_MC_loop, total_histories, num_p, num_projections, D_angle, initial_angle);
		if (!report_image_to_array(ns_ptr, cp_ptr, rl_ptr, ms_ptr)) {
			mexPrintf("Failed to report image to array at view %d!\n", num_p);
			return false;
		}
		ns_ptr += pixel_num;
		cp_ptr += pixel_num;
		rl_ptr += pixel_num;
		ms_ptr += pixel_num;
		ioFlush();
		// *** Clear the image after reporting, unless this is the last projection to simulate:
		if (num_p < (num_projections - 1))
			init_image_array_GPU(image_device, pixel_num);

	} // [Projection loop end: iterate for next CT projection angle]

	///////////////////////////////////////////////////////////////////////////////////////////////////

	// *** Simulation finished! Report dose and timings and clean up.
	if (dose_ROI_x_max > -1)
	{
		clock_kernel = clock();

		SAFE_CALL(cudaMemcpy(voxels_Edep, voxels_Edep_device, voxels_Edep_bytes, cudaMemcpyDeviceToHost)); // Copy final dose results to host (for every MPI threads)

		mexPrintf("       ==> CUDA: Time copying dose results from device to host: %.6f s\n", float(clock() - clock_kernel) / CLOCKS_PER_SEC);
	}

	if (flag_material_dose == 1)
		SAFE_CALL(cudaMemcpy(materials_dose, materials_dose_device, MAX_MATERIALS * sizeof(ulonglong2), cudaMemcpyDeviceToHost)); // Copy materials dose results to host, if tally enabled in input file.   !!tally_materials_dose!!

	// *** Report the total dose for all the projections, if the tally is not disabled (must be done after MPI_Barrier to have all the MPI threads synchronized):
	clock_start = clock();

	if (dose_ROI_x_max > -1)
	{
		// -- Report the total dose for all the projections:
		report_voxels_dose(file_dose_output, num_projections, &voxel_data, voxel_mat_dens, voxels_Edep, time_total_MC_simulation, total_histories, dose_ROI_x_min, dose_ROI_x_max, dose_ROI_y_min, dose_ROI_y_max, dose_ROI_z_min, dose_ROI_z_max, source_data);
	}

	// -- Report "tally_materials_dose" with data from all MPI threads, if tally enabled:
	if (flag_material_dose == 1)
	{
		ulonglong2 *materials_dose_total = materials_dose; // Create a dummy pointer to the materials_dose data
		report_materials_dose(num_projections, total_histories, density_nominal, materials_dose_total, mass_materials); // Report the material dose  !!tally_materials_dose!!
	}

	clock_end = clock();
	mexPrintf("\n\n       ==> CUDA: Time reporting the dose data: %.6f s\n", ((double)(clock_end - clock_start)) / CLOCKS_PER_SEC);

	_print_finish(clock_start_beginning, time_total_MC_simulation);
	CHECK_AND_DELETE(voxel_mat_dens);
	ioFlush();
	CHECK_AND_DELETE(mfp_Woodcock_table);
	CHECK_AND_DELETE(mfp_table_a);
	CHECK_AND_DELETE(mfp_table_b);
	CHECK_AND_FREE(voxel_mat_dens_device);
	CHECK_AND_FREE(image_device);
	CHECK_AND_FREE(mfp_Woodcock_table_device);
	CHECK_AND_FREE(mfp_table_a_device);
	CHECK_AND_FREE(mfp_table_b_device);
	CHECK_AND_FREE(voxels_Edep_device);
	return true;
}

bool MCGPU::clear()
{
	// *** Clean up RAM memory. If CUDA was used, the geometry and table data were already cleaned for MPI threads other than root after copying data to the GPU:
	CHECK_AND_DELETE(voxels_Edep);
	CHECK_AND_DELETE(image);
	CHECK_AND_DELETE(mfp_Woodcock_table);
	CHECK_AND_DELETE(mfp_table_a);
	CHECK_AND_DELETE(mfp_table_b);
	CHECK_AND_DELETE(voxel_mat_dens);
	CHECK_AND_FREE(voxel_mat_dens_device);
	CHECK_AND_FREE(image_device);
	CHECK_AND_FREE(mfp_Woodcock_table_device);
	CHECK_AND_FREE(mfp_table_a_device);
	CHECK_AND_FREE(mfp_table_b_device);
	CHECK_AND_FREE(voxels_Edep_device);
	// cudaDeviceReset(); // Destroy the CUDA context before ending program (flush visual debugger data).
	return true;
}

size_t MCGPU::getProjWidth()
{
	return detector_data[0].num_pixels.x;
}

size_t MCGPU::getProjHeight()
{
	return detector_data[0].num_pixels.y;
}

size_t MCGPU::getProjNum()
{
	return num_projections;
}

#define GET_FIELD(name) do {				\
	array = mxGetField(param, 0, name);		\
	if (NULL == array) {					\
		mexPrintf("%s not found!", name);	\
		return false;						\
	}										\
} while(0)

bool MCGPU::parse_param_struct(mxArray* param)
{
	mxArray* array;
	double* ptr1;
	GET_FIELD("total_histories");
	total_histories = static_cast<unsigned long long>(mxGetScalar(array));
	GET_FIELD("seed_input");
	seed_input = static_cast<int>(mxGetScalar(array));
	GET_FIELD("gpu_id");
	gpu_id = static_cast<int>(mxGetScalar(array));
	GET_FIELD("num_threads_per_block");
	num_threads_per_block = static_cast<int>(mxGetScalar(array));
	GET_FIELD("histories_per_thread");
	histories_per_thread = static_cast<int>(mxGetScalar(array));
	GET_FIELD("spec");
	if (!parse_spec_struct(array)) {
		mexErrMsgTxt("Failed to parse spectrum!");
		return false;
	}
	GET_FIELD("source_pos");
	ptr1 = reinterpret_cast<double*>(mxGetData(array));
	source_data[0].position.x = ptr1[0];
	source_data[0].position.y = ptr1[1];
	source_data[0].position.z = ptr1[2];
	GET_FIELD("source_dir");
	ptr1 = reinterpret_cast<double*>(mxGetData(array));
	source_data[0].direction.x = ptr1[0];
	source_data[0].direction.y = ptr1[1];
	source_data[0].direction.z = ptr1[2];
	GET_FIELD("aperture");
	ptr1 = reinterpret_cast<double*>(mxGetData(array));
	double phi_aperture = ptr1[0];
	double theta_aperture = ptr1[1];
	// *** RECTANGULAR BEAM INITIALIZATION: aperture initially centered at (0,1,0), ie, THETA_0=90, PHI_0=90
    //     Using the algorithm used in PENMAIN.f, from penelope 2008 (by F. Salvat).
    source_data[0].cos_theta_low = (float)(cos((90.0 - 0.5 * theta_aperture) * DEG2RAD));
    source_data[0].D_cos_theta = (float)(-2.0 * source_data[0].cos_theta_low); // Theta aperture is symetric above and below 90 deg
    source_data[0].phi_low = (float)((90.0 - 0.5 * phi_aperture) * DEG2RAD);
    source_data[0].D_phi = (float)(phi_aperture * DEG2RAD);
    source_data[0].max_height_at_y1cm = (float)(tan(0.5 * theta_aperture * DEG2RAD));

    // If a pencil beam is input, convert the 0 angle to a very small square beam to avoid precission errors:
    if (fabs(theta_aperture) < 1.0e-7)
    {
        theta_aperture = +1.00e-7;
        source_data[0].cos_theta_low = 0.0f; // = cos(90-0)
        source_data[0].D_cos_theta = 0.0f;
        source_data[0].max_height_at_y1cm = 0.0f;
    }
    if (fabs(phi_aperture) < 1.0e-7)
    {
        phi_aperture = +1.00e-7;
        source_data[0].phi_low = (float)(90.0 * DEG2RAD);
        source_data[0].D_phi = 0.0f;
    }

	GET_FIELD("output_proj_type");
	output_proj_type = static_cast<int>(mxGetScalar(array));
	GET_FIELD("file_name_output");
	mxGetString(array, file_name_output, 250);
	GET_FIELD("dummy_num_pixels");
	int32_t* ptr2 = reinterpret_cast<int32_t*>(mxGetData(array));
	detector_data[0].num_pixels.x = ptr2[0];
	detector_data[0].num_pixels.y = ptr2[1];
	detector_data[0].total_num_pixels = detector_data[0].num_pixels.x * detector_data[0].num_pixels.y;
	GET_FIELD("det_size");
	ptr1 = reinterpret_cast<double*>(mxGetData(array));
	detector_data[0].width_X = ptr1[0];
	detector_data[0].height_Z = ptr1[1];
	detector_data[0].inv_pixel_size_X = detector_data[0].num_pixels.x / detector_data[0].width_X;
    detector_data[0].inv_pixel_size_Z = detector_data[0].num_pixels.y / detector_data[0].height_Z;
	GET_FIELD("sdd");
	detector_data[0].sdd = static_cast<float>(mxGetScalar(array));
	float3 detector_center; // Center of the detector straight ahead of the focal spot.
    detector_center.x = source_data[0].position.x + source_data[0].direction.x * detector_data[0].sdd;
    detector_center.y = source_data[0].position.y + source_data[0].direction.y * detector_data[0].sdd;
    detector_center.z = source_data[0].position.z + source_data[0].direction.z * detector_data[0].sdd;
	if ((theta_aperture < -1.0e-7) || (phi_aperture < -1.0e-7)) // If we enter a negative angle, the fan beam will cover exactly the detector surface.
    {
        theta_aperture = 2.0 * atan(0.5 * detector_data[0].height_Z / (detector_data[0].sdd)) * RAD2DEG; // Optimum angles
        phi_aperture = 2.0 * atan(0.5 * detector_data[0].width_X / (detector_data[0].sdd)) * RAD2DEG;

        source_data[0].cos_theta_low = (float)(cos((90.0 - 0.5 * theta_aperture) * DEG2RAD));
        source_data[0].D_cos_theta = (float)(-2.0 * source_data[0].cos_theta_low); // Theta aperture is symetric above and below 90 deg
        source_data[0].phi_low = (float)((90.0 - 0.5 * phi_aperture) * DEG2RAD);
        source_data[0].D_phi = (float)(phi_aperture * DEG2RAD);
        source_data[0].max_height_at_y1cm = (float)(tan(0.5 * theta_aperture * DEG2RAD));
    }
	GET_FIELD("num_projections");
	num_projections = static_cast<int>(mxGetScalar(array));
	if ((num_projections > 1) && (fabs(source_data[0].direction.z) > 0.00001f))
    {
        mexPrintf("\n\n   !!read_input ERROR!! Sorry, but currently we can only simulate CT scans when the source direction is perpendicular to the Z axis (ie, w=0).\n\n\n"); // The reconstructed planes are always parallel to the XY plane.\n");
        return false;
    }
	if (num_projections > MAX_NUM_PROJECTIONS)
    {
        mexPrintf("\n\n   !!read_input ERROR!! The input number of projections is too large. Increase parameter MAX_NUM_PROJECTIONS=%d in the header file and recompile.\n", MAX_NUM_PROJECTIONS);
        mexPrintf("                        There is no limit in the number of projections to be simulated because the source, detector data for each projection is stored in global memory and transfered to shared memory for each projection.\n\n");
        return false;
    }
	if (num_projections > 1) {
		GET_FIELD("D_angle");
		D_angle = static_cast<float>(mxGetScalar(array)) * DEG2RAD;
		// Calculate initial source angle:
        initial_angle = acos((double)(source_data[0].direction.x));
        if (source_data[0].direction.y < 0)
            initial_angle = -(initial_angle); // Correct for the fact that positive and negative angles have the same ACOS
        if (initial_angle < 0.0)
            initial_angle = (initial_angle) + 2.0 * PI; // Make sure the angle is not negative, between [0,360) degrees.
        initial_angle = (initial_angle) - PI;           // Correct the fact that the source is opposite to the detector (180 degrees difference).
        if (initial_angle < 0.0)
            initial_angle = (initial_angle) + 2.0 * PI; // Make sure the initial angle is not negative, between [0,360) degrees.
		GET_FIELD("angularROI");
		ptr1 = reinterpret_cast<double*>(mxGetData(array));
		angularROI_0 = ptr1[0];
		angularROI_1 = ptr1[1];
		angularROI_0 = (angularROI_0 - 0.00001) * DEG2RAD; // Store the angles of interest in radians, increasing a little the interval to avoid floating point precision problems
        angularROI_1 = (angularROI_1 + 0.00001) * DEG2RAD;
		GET_FIELD("sod");
		SRotAxisD = static_cast<double>(mxGetScalar(array));
		GET_FIELD("vertical_translation_per_projection");
		vertical_translation_per_projection = static_cast<double>(mxGetScalar(array));
	}
	GET_FIELD("flag_material_dose");
	flag_material_dose = static_cast<int>(mxGetScalar(array));
	GET_FIELD("tally_3D_dose");
	int flag = static_cast<int>(mxGetScalar(array));
	if (flag) {
		GET_FIELD("file_dose_output");
		mxGetString(array, file_dose_output, 250);
		GET_FIELD("dose_ROI_x");
		ptr1 = reinterpret_cast<double*>(mxGetData(array));
		dose_ROI_x_min = ptr1[0];
		dose_ROI_x_max = ptr1[1];
		GET_FIELD("dose_ROI_y");
		ptr1 = reinterpret_cast<double*>(mxGetData(array));
		dose_ROI_y_min = ptr1[0];
		dose_ROI_y_max = ptr1[1];
		GET_FIELD("dose_ROI_z");
		ptr1 = reinterpret_cast<double*>(mxGetData(array));
		dose_ROI_z_min = ptr1[0];
		dose_ROI_z_max = ptr1[1];
		mexPrintf("       3D voxel dose deposition tally ENABLED.\n");
        if ((dose_ROI_x_min > dose_ROI_x_max) || (dose_ROI_y_min > dose_ROI_y_max) || (dose_ROI_z_min > dose_ROI_z_max) ||
            dose_ROI_x_min < 0 || dose_ROI_y_min < 0 || dose_ROI_z_min < 0)
        {
            mexPrintf("\n\n   !!read_input ERROR!! The input region-of-interst in \'SECTION DOSE DEPOSITION\' is not valid: the minimum voxel index may not be zero or larger than the maximum index.\n");
            printf("                          Input data = X[%d,%d], Y[%d,%d], Z[%d,%d]\n\n", dose_ROI_x_min, dose_ROI_x_max, dose_ROI_y_min, dose_ROI_y_max, dose_ROI_z_min, dose_ROI_z_max); // Show ROI with index=1 for the first voxel instead of 0.
            return false;
        }
        if ((dose_ROI_x_min == dose_ROI_x_max) && (dose_ROI_y_min == dose_ROI_y_max) && (dose_ROI_z_min == dose_ROI_z_max))
        {
            mexPrintf("\n\n   !!read_input!! According to the input region-of-interest in \'SECTION DOSE DEPOSITION\', only the dose in the voxel (%d,%d,%d) will be tallied.\n\n", dose_ROI_x_min, dose_ROI_y_min, dose_ROI_z_min);
        }
	} else {
		// -- NO: disabling tally
        mexPrintf("       3D voxel dose deposition tally DISABLED.\n");
        dose_ROI_x_min = (short int)32500;
        dose_ROI_x_max = (short int)-32500; // Set absurd values for the ROI to make sure we never get any dose tallied
        dose_ROI_y_min = (short int)32500;
        dose_ROI_y_max = (short int)-32500; // (the maximum values for short int variables are +-32768)
        dose_ROI_z_min = (short int)32500;
        dose_ROI_z_max = (short int)-32500;
	}
	GET_FIELD("file_name_voxels");
	mxGetString(array, file_name_voxels, 250);
	GET_FIELD("file_name_materials");
	for (int i = 0; i < MAX_MATERIALS; ++i) {
		mxArray* tmp = mxGetCell(array, i);
		mxGetString(tmp, file_name_materials[i], 250);
	}
	// finish parse parameters
	// *** Set the rotation that will bring particles from the detector plane to +Y=(0,+1,0) through a rotation around X and around Z (counter-clock):
    double rotX, rotZ, cos_rX, cos_rZ, sin_rX, sin_rZ;
    // rotX = 1.5*PI - acos(source_data[0].direction.z);  // Rotate to +Y = (0,+1,0) --> rotX_0 = 3/2*PI == -PI/2
    rotX = acos(source_data[0].direction.z) - 0.5 * PI; // Rotate to +Y = (0,+1,0) --> rotX_0 =  -PI/2
                                                        // rotX = 0.5*PI - acos(source_data[0].direction.z);  // Rotate to +Y = (0,+1,0) --> rotX_0 =  PI/2
    if ((source_data[0].direction.x * source_data[0].direction.x + source_data[0].direction.y * source_data[0].direction.y) > 1.0e-8) // == u^2+v^2 > 0
    {
        // rotZ = 0.5*PI - acos(source_data[0].direction.x/sqrt(source_data[0].direction.x*source_data[0].direction.x + source_data[0].direction.y*source_data[0].direction.y));
        if (source_data[0].direction.y >= 0.0f)
            rotZ = 0.5 * PI - acos(source_data[0].direction.x / sqrt(source_data[0].direction.x * source_data[0].direction.x + source_data[0].direction.y * source_data[0].direction.y));
        else
            rotZ = 0.5 * PI - (-acos(source_data[0].direction.x / sqrt(source_data[0].direction.x * source_data[0].direction.x + source_data[0].direction.y * source_data[0].direction.y)));
    }
    else
        rotZ = 0.0; // Vector pointing to +Z, do not rotate around Z then.

    // -- Set the rotation matrix RzRx (called inverse because moves from the correct position to the reference at +Y):
    cos_rX = cos(rotX);
    cos_rZ = cos(rotZ);
    sin_rX = sin(rotX);
    sin_rZ = sin(rotZ);

    // Rotation matrix RxRz:
    detector_data[0].rot_inv[0] = cos_rZ;
    detector_data[0].rot_inv[1] = -sin_rZ;
    detector_data[0].rot_inv[2] = 0.0f;
    detector_data[0].rot_inv[3] = cos_rX * sin_rZ;
    detector_data[0].rot_inv[4] = cos_rX * cos_rZ;
    detector_data[0].rot_inv[5] = -sin_rX;
    detector_data[0].rot_inv[6] = sin_rX * sin_rZ;
    detector_data[0].rot_inv[7] = sin_rX * cos_rZ;
    detector_data[0].rot_inv[8] = cos_rX;

    if ((source_data[0].direction.y > 0.99999f) && (num_projections == 1))
    {
        // Simulating a single projection and initial beam pointing to +Y: no rotation needed!!
        detector_data[0].rotation_flag = 0;
        detector_data[0].corner_min_rotated_to_Y.x = detector_center.x;
        detector_data[0].corner_min_rotated_to_Y.y = detector_center.y;
        detector_data[0].corner_min_rotated_to_Y.z = detector_center.z;

        mexPrintf("       Source pointing to (0,1,0): detector not rotated, initial location in voxels found faster.\n"); // maximizing code efficiency -> the simulation will be faster than for other angles (but not much).");
    }
    else
    { // Rotation needed to set the detector perpendicular to +Y:
        detector_data[0].rotation_flag = 1;
        // -- Rotate the detector center to +Y:
        detector_data[0].corner_min_rotated_to_Y.x = detector_center.x * detector_data->rot_inv[0] + detector_center.y * detector_data[0].rot_inv[1] + detector_center.z * detector_data[0].rot_inv[2];
        detector_data[0].corner_min_rotated_to_Y.y = detector_center.x * detector_data[0].rot_inv[3] + detector_center.y * detector_data[0].rot_inv[4] + detector_center.z * detector_data[0].rot_inv[5];
        detector_data[0].corner_min_rotated_to_Y.z = detector_center.x * detector_data[0].rot_inv[6] + detector_center.y * detector_data[0].rot_inv[7] + detector_center.z * detector_data[0].rot_inv[8];

        mexPrintf("       Rotations from the input direction to +Y [deg]: rotZ = %f , rotX = %f\n", rotZ * RAD2DEG, rotX * RAD2DEG);
    }
    // -- Set the lower corner (minimum) coordinates at the normalized orientation: +Y. The detector has thickness 0.
    detector_data[0].corner_min_rotated_to_Y.x = detector_data[0].corner_min_rotated_to_Y.x - 0.5 * detector_data[0].width_X;
    //  detector_data[0].corner_min_rotated_to_Y.y = detector_data[0].corner_min_rotated_to_Y.y;
    detector_data[0].corner_min_rotated_to_Y.z = detector_data[0].corner_min_rotated_to_Y.z - 0.5 * detector_data[0].height_Z;

    detector_data[0].center.x = source_data[0].position.x + source_data[0].direction.x * detector_data[0].sdd;
    detector_data[0].center.y = source_data[0].position.y + source_data[0].direction.y * detector_data[0].sdd;
    detector_data[0].center.z = source_data[0].position.z + source_data[0].direction.z * detector_data[0].sdd;

	// *** Init the fan beam source model:
    if (1 == detector_data[0].rotation_flag)
    {
        // Initial beam NOT pointing to +Y: rotation is needed to move the sampled vector from (0,1,0) to the given direction!!
        rotX = 0.5 * PI - acos(source_data[0].direction.z);                              // ! Rotation about X: acos(wsrc)==theta, theta=90 for alpha=0, ie, +Y.
        rotZ = atan2(source_data[0].direction.y, source_data[0].direction.x) - 0.5 * PI; // ! Rotation about Z:  initial phi = 90 (+Y).  [ATAN2(v,u) = TAN(v/u), with the angle in the correct quadrant.
        cos_rX = cos(rotX);
        cos_rZ = cos(rotZ);
        sin_rX = sin(rotX);
        sin_rZ = sin(rotZ);
        // --Rotation around X (alpha) and then around Z (phi): Rz*Rx (oposite of detector rotation)
        source_data[0].rot_fan[0] = cos_rZ;
        source_data[0].rot_fan[1] = -cos_rX * sin_rZ;
        source_data[0].rot_fan[2] = sin_rX * sin_rZ;
        source_data[0].rot_fan[3] = sin_rZ;
        source_data[0].rot_fan[4] = cos_rX * cos_rZ;
        source_data[0].rot_fan[5] = -sin_rX * cos_rZ;
        source_data[0].rot_fan[6] = 0.0f;
        source_data[0].rot_fan[7] = sin_rX;
        source_data[0].rot_fan[8] = cos_rX;

        mexPrintf("       Rotations from +Y to the input direction for the fan beam source model [deg]: rotZ = %f , rotX = %f\n", rotZ * RAD2DEG, rotX * RAD2DEG);
    }

	// *** Allocate array for the 4 detected images (non-scattered, Compton, Rayleigh, multiple-scatter):
    int pixels_per_image = detector_data[0].num_pixels.x * detector_data[0].num_pixels.y;
    image_bytes = 4 * pixels_per_image * sizeof(unsigned long long int);
    (image) = (unsigned long long int *)malloc(image_bytes);
    if (image == NULL)
    {
        mexPrintf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d pixels for the 4 scatter images (%f Mbytes)!!\n\n", pixels_per_image, (image_bytes) / (1024.f * 1024.f));
        return false;
    } else
    {
        mexPrintf("       Array for 4 scatter images correctly allocated (%d pixels, %f Mbytes)\n", pixels_per_image, (image_bytes) / (1024.f * 1024.f));
    }

	// *** Allocate dose and dose^2 array if tally active:
    int num_voxels_ROI = ((int)(dose_ROI_x_max - dose_ROI_x_min + 1)) * ((int)(dose_ROI_y_max - dose_ROI_y_min + 1)) * ((int)(dose_ROI_z_max - dose_ROI_z_min + 1));
    if (dose_ROI_x_max > -1)
    {
        voxels_Edep_bytes = num_voxels_ROI * sizeof(ulonglong2);
        voxels_Edep = (ulonglong2 *)malloc(voxels_Edep_bytes);
        if (voxels_Edep == NULL)
        {
            mexPrintf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d voxels for the deposited dose (and uncertainty) array (%f Mbytes)!!\n\n", num_voxels_ROI, voxels_Edep_bytes / (1024.f * 1024.f));
            return false;
        } else
        {
            mexPrintf("       Array for the deposited dose ROI (and uncertainty) correctly allocated (%d voxels, %f Mbytes)\n", num_voxels_ROI, voxels_Edep_bytes / (1024.f * 1024.f));
        }
    }
    else
    {
        voxels_Edep_bytes = 0;
    }
	// *** Initialize the voxel dose to 0 in the CPU. Not necessary for the CUDA code if dose matrix init. in the GPU global memory using a GPU kernel, but needed if using cudaMemcpy.
    if (dose_ROI_x_max > -1)
    {
        memset(voxels_Edep, 0, voxels_Edep_bytes); // Init memory space to 0.
    }
	return true;
}

bool MCGPU::parse_spec_struct(mxArray* param)
{
	mxArray* array;
	GET_FIELD("prob_espc_bin");
	float* prob_spec_bin = reinterpret_cast<float*>(mxGetData(array));
	GET_FIELD("espc");
	float* espc = reinterpret_cast<float*>(mxGetData(array));
	GET_FIELD("num_bins_espc");
	source_energy_data.num_bins_espc = static_cast<int>(mxGetScalar(array))-1;
	GET_FIELD("mean_energy_spectrum");
	mean_energy_spectrum = static_cast<float>(mxGetScalar(array));
	for (int i = 0; i < MAX_ENERGY_BINS; ++i) {
		source_energy_data.espc[i] = espc[i];
	}
	IRND0(prob_spec_bin, source_energy_data.espc_cutoff, source_energy_data.espc_alias, source_energy_data.num_bins_espc);
	return true;
}

bool MCGPU::parse_data_struct(mxArray* param)
{
	mxArray* array;
	double* ptr1;
	GET_FIELD("num_voxels");
	ptr1 = reinterpret_cast<double*>(mxGetData(array));
	voxel_data.num_voxels.x = ptr1[0];
	voxel_data.num_voxels.y = ptr1[1];
	voxel_data.num_voxels.z = ptr1[2];
	GET_FIELD("voxel_size");
	ptr1 = reinterpret_cast<double*>(mxGetData(array));
	GET_FIELD("material");
	float* material = reinterpret_cast<float*>(mxGetData(array));
	GET_FIELD("density");
	float* density = reinterpret_cast<float*>(mxGetData(array));
	GET_FIELD("density_max");
	float* dmax = reinterpret_cast<float*>(mxGetData(array));
	memcpy(density_max, dmax, MAX_MATERIALS*sizeof(float));
	// -- Store the size of the voxel bounding box (used in the source function):
  	voxel_data.size_bbox.x = voxel_data.num_voxels.x * ptr1[0];
  	voxel_data.size_bbox.y = voxel_data.num_voxels.y * ptr1[1];
  	voxel_data.size_bbox.z = voxel_data.num_voxels.z * ptr1[2];
	mexPrintf("       Number of voxels in the input geometry file: %d x %d x %d =  %d\n", voxel_data.num_voxels.x, voxel_data.num_voxels.y, voxel_data.num_voxels.z, (voxel_data.num_voxels.x * voxel_data.num_voxels.y * voxel_data.num_voxels.z));
	mexPrintf("       Size of the input voxels: %f x %f x %f cm  (voxel volume=%f cm^3)\n", ptr1[0], ptr1[1], ptr1[2], ptr1[0]*ptr1[1]*ptr1[2]);
	mexPrintf("       Voxel bounding box size: %f x %f x %f cm\n", voxel_data.size_bbox.x, voxel_data.size_bbox.y, voxel_data.size_bbox.z);

	if ((dose_ROI_x_max + 1) > (voxel_data.num_voxels.x) || (dose_ROI_y_max + 1) > (voxel_data.num_voxels.y) || (dose_ROI_z_max + 1) > (voxel_data.num_voxels.z))
	{
		mexPrintf("\n       The input region of interest for the dose deposition is larger than the size of the voxelized geometry:\n");
		dose_ROI_x_max = min_value(voxel_data.num_voxels.x - 1, dose_ROI_x_max);
		dose_ROI_y_max = min_value(voxel_data.num_voxels.y - 1, dose_ROI_y_max);
		dose_ROI_z_max = min_value(voxel_data.num_voxels.z - 1, dose_ROI_z_max);
		mexPrintf("       updating the ROI max limits to fit the geometry -> dose_ROI_max=(%d, %d, %d)\n", dose_ROI_x_max + 1, dose_ROI_y_max + 1, dose_ROI_z_max + 1); // Allowing the input of an ROI larger than the voxel volume: in this case some of the allocated memory will be wasted but the program will run ok.
	}
	if ((dose_ROI_x_max + 1) == (voxel_data.num_voxels.x) && (dose_ROI_y_max + 1) == (voxel_data.num_voxels.y) && (dose_ROI_z_max + 1) == (voxel_data.num_voxels.z))
		mexPrintf("       The voxel dose tally ROI covers the entire voxelized phantom: the dose to every voxel will be tallied.\n");
	else
		mexPrintf("       The voxel dose tally ROI covers only a fraction of the voxelized phantom: the dose to voxels outside the ROI will not be tallied.\n");

	// -- Store the inverse of the pixel sides (in cm) to speed up the particle location in voxels.
  	voxel_data.inv_voxel_size.x = 1.0f/(ptr1[0]);
  	voxel_data.inv_voxel_size.y = 1.0f/(ptr1[1]);
  	voxel_data.inv_voxel_size.z = 1.0f/(ptr1[2]);
	// -- Allocate the voxel matrix and store array size:
	size_t voxel_num = voxel_data.num_voxels.x * voxel_data.num_voxels.y * voxel_data.num_voxels.z;
	voxel_mat_dens_bytes = sizeof(float2) * voxel_num;
	CHECK_AND_DELETE(voxel_mat_dens);
	voxel_mat_dens = (float2*) malloc(voxel_mat_dens_bytes);
	if (voxel_mat_dens == NULL)
	{
		mexPrintf("\n\n   !!malloc ERROR load_voxels!! Not enough memory to allocate %d voxels (%f Mbytes)!!\n\n", (voxel_data.num_voxels.x * voxel_data.num_voxels.y * voxel_data.num_voxels.z), voxel_mat_dens_bytes / (1024.f * 1024.f));
		return false;
	}

	mexPrintf("\n    -- Initializing the voxel material and density vector (%f Mbytes)...\n", voxel_mat_dens_bytes / (1024.f * 1024.f));

	// -- Read the voxel densities:
	float2 *voxels_ptr = voxel_mat_dens;
	for (int i = 0; i < voxel_num; ++i)
	{
		if (material[i] > MAX_MATERIALS)
		{
			mexPrintf("\n\n   !!ERROR load_voxels!! Voxel material number too high!! #mat=%d, MAX_MATERIALS=%d, voxel number=%d\n\n", material[i], MAX_MATERIALS, i);
			return false;
		}
		if (material[i] < 1)
		{
			mexPrintf("\n\n   !!ERROR load_voxels!! Voxel material number can not be zero or negative!! #mat=%d, voxel number=%dd\n\n", material[i], i);
			return false;
		}
		if (density[i] < 1.0e-9f)
		{
			mexPrintf("\n\n   !!ERROR load_voxels!! Voxel density can not be 0 or negative: #mat=%d, density=%f, voxel number=%d\n\n", material[i], density[i], i);
			return false;
		}

		(*voxels_ptr).x = (float)(material[i]) + 0.0001f; // Assign material value as float (the integer value will be recovered by truncation)
		(*voxels_ptr).y = density[i];					 // Assign density value
		voxels_ptr++; // Move to next voxel
	}
	return true;
}

bool MCGPU::set_CT_trajectory(int num_projections, double D_angle,
							  double angularROI_0, double angularROI_1, 
							  double SRotAxisD,source_struct *source_data, 
							  detector_struct *detector_data,
							  double vertical_translation_per_projection)
{
	mexPrintf("\n    -- Setting the sources and detectors for the %d CT projections (MAX_NUM_PROJECTIONS=%d):\n", num_projections, MAX_NUM_PROJECTIONS);
	double cos_rX, cos_rZ, sin_rX, sin_rZ, current_angle;

	// --Set center of rotation at the input distance between source and detector:
	float3 center_rotation;
	center_rotation.x = source_data[0].position.x + source_data[0].direction.x * SRotAxisD;
	center_rotation.y = source_data[0].position.y + source_data[0].direction.y * SRotAxisD;
	center_rotation.z = source_data[0].position.z; //  + source_data[0].direction.z * SRotAxisD;   // w=0 all the time!!

	// --Angular span between projections:

	//  -Set initial angle for the source (180 degress less than the detector pointed by the direction vector; the zero angle is the X axis, increasing to +Y axis).
	current_angle = acos((double)source_data[0].direction.x);
	if (source_data[0].direction.y < 0)
		current_angle = -current_angle; // Correct for the fact that positive and negative angles have the same ACOS
	if (current_angle < 0.0)
		current_angle += 2.0 * PI;		// Make sure the angle is not negative, between [0,360) degrees.
	current_angle = current_angle - PI; // Correct the fact that the source is opposite to the detector (180 degrees difference).
	if (current_angle < 0.0)
		current_angle += 2.0 * PI; // Make sure the angle is not negative, between [0,360) degrees..

	mexPrintf("         << Projection #1 >> initial_angle=%.8f , D_angle=%.8f\n", current_angle * RAD2DEG, D_angle * RAD2DEG);
	mexPrintf("                             Source direction=(%.8f,%.8f,%.8f), position=(%.8f,%.8f,%.8f)\n", source_data[0].direction.x, source_data[0].direction.y, source_data[0].direction.z, source_data[0].position.x, source_data[0].position.y, source_data[0].position.z);

	for (int i = 1; i < num_projections; i++) // The first projection (i=0) was initialized in function "read_input".
	{
		// --Init constant parameters to the values in projection 0:
		source_data[i].cos_theta_low = source_data[0].cos_theta_low;
		source_data[i].phi_low = source_data[0].phi_low;
		source_data[i].D_cos_theta = source_data[0].D_cos_theta;
		source_data[i].D_phi = source_data[0].D_phi;
		source_data[i].max_height_at_y1cm = source_data[0].max_height_at_y1cm;
		detector_data[i].sdd = detector_data[0].sdd;
		detector_data[i].width_X = detector_data[0].width_X;
		detector_data[i].height_Z = detector_data[0].height_Z;
		detector_data[i].inv_pixel_size_X = detector_data[0].inv_pixel_size_X;
		detector_data[i].inv_pixel_size_Z = detector_data[0].inv_pixel_size_Z;
		detector_data[i].num_pixels = detector_data[0].num_pixels;
		detector_data[i].total_num_pixels = detector_data[0].total_num_pixels;
		detector_data[i].rotation_flag = detector_data[0].rotation_flag;

		// --Set the new source location and direction, for the current CT projection:
		current_angle += D_angle;
		if (current_angle >= (2.0 * PI - 0.0001))
			current_angle -= 2.0 * PI; // Make sure the angle is not above or equal to 360 degrees.

		source_data[i].position.x = center_rotation.x + SRotAxisD * cos(current_angle);
		source_data[i].position.y = center_rotation.y + SRotAxisD * sin(current_angle);
		source_data[i].position.z = source_data[i - 1].position.z + vertical_translation_per_projection; //  The Z position can increase between projections for a helical scan. But rotation still around Z always: (w=0)!!

		source_data[i].direction.x = center_rotation.x - source_data[i].position.x;
		source_data[i].direction.y = center_rotation.y - source_data[i].position.y;
		source_data[i].direction.z = 0.0f; //  center_rotation.z - source_data[0].position.z;   !! w=0 all the time!!

		double norm = 1.0 / sqrt((double)source_data[i].direction.x * (double)source_data[i].direction.x + (double)source_data[i].direction.y * (double)source_data[i].direction.y /* + source_data[i].direction.z*source_data[i].direction.z*/);
		source_data[i].direction.x = (float)(((double)source_data[i].direction.x) * norm);
		source_data[i].direction.y = (float)(((double)source_data[i].direction.y) * norm);
		// source_data[i].direction.z = (float)(((double)source_data[i].direction.z)*norm);

		// --Set the new detector in front of the new source:
		detector_data[i].center.x = source_data[i].position.x + source_data[i].direction.x * detector_data[i].sdd; // Set the center of the detector straight ahead of the focal spot.
		detector_data[i].center.y = source_data[i].position.y + source_data[i].direction.y * detector_data[i].sdd;
		detector_data[i].center.z = source_data[i].position.z; //  + source_data[i].direction.z * detector_data[i].sdd;   !! w=0 all the time!!

		double rotX, rotZ;

		//  detector_data[0].rotation_flag = 1;   //  Already set in read_input!

		// -- Rotate the detector center to +Y:
		//    Set the rotation that will bring particles from the detector plane to +Y=(0,+1,0) through a rotation around X and around Z (counter-clock):
		rotX = 0.0; // !! w=0 all the time!!  CORRECT CALCULATION:  acos(source_data[0].direction.z) - 0.5*PI;  // Rotate to +Y = (0,+1,0) --> rotX_0 =  -PI/2

		if ((source_data[i].direction.x * source_data[i].direction.x + source_data[i].direction.y * source_data[i].direction.y) > 1.0e-8) // == u^2+v^2 > 0
			if (source_data[i].direction.y >= 0.0f)
				rotZ = 0.5 * PI - acos(source_data[i].direction.x / sqrt(source_data[i].direction.x * source_data[i].direction.x + source_data[i].direction.y * source_data[i].direction.y));
			else
				rotZ = 0.5 * PI - (-acos(source_data[i].direction.x / sqrt(source_data[i].direction.x * source_data[i].direction.x + source_data[i].direction.y * source_data[i].direction.y)));
		else
			rotZ = 0.0; // Vector pointing to +Z, do not rotate around Z then.

		mexPrintf("         << Projection #%d >> current_angle=%.8f degrees (rotation around Z axis = %.8f)\n", (i + 1), current_angle * RAD2DEG, rotZ * RAD2DEG);
		mexPrintf("                             Source direction = (%.8f,%.8f,%.8f) , position = (%.8f,%.8f,%.8f)\n", source_data[i].direction.x, source_data[i].direction.y, source_data[i].direction.z, source_data[i].position.x, source_data[i].position.y, source_data[i].position.z);

		cos_rX = cos(rotX);
		cos_rZ = cos(rotZ);
		sin_rX = sin(rotX);
		sin_rZ = sin(rotZ);
		detector_data[i].rot_inv[0] = cos_rZ; // Rotation matrix RxRz:
		detector_data[i].rot_inv[1] = -sin_rZ;
		detector_data[i].rot_inv[2] = 0.0f;
		detector_data[i].rot_inv[3] = cos_rX * sin_rZ;
		detector_data[i].rot_inv[4] = cos_rX * cos_rZ;
		detector_data[i].rot_inv[5] = -sin_rX;
		detector_data[i].rot_inv[6] = sin_rX * sin_rZ;
		detector_data[i].rot_inv[7] = sin_rX * cos_rZ;
		detector_data[i].rot_inv[8] = cos_rX;

		detector_data[i].corner_min_rotated_to_Y.x = detector_data[i].center.x * detector_data[i].rot_inv[0] + detector_data[i].center.y * detector_data[i].rot_inv[1] + detector_data[i].center.z * detector_data[i].rot_inv[2];
		detector_data[i].corner_min_rotated_to_Y.y = detector_data[i].center.x * detector_data[i].rot_inv[3] + detector_data[i].center.y * detector_data[i].rot_inv[4] + detector_data[i].center.z * detector_data[i].rot_inv[5];
		detector_data[i].corner_min_rotated_to_Y.z = detector_data[i].center.x * detector_data[i].rot_inv[6] + detector_data[i].center.y * detector_data[i].rot_inv[7] + detector_data[i].center.z * detector_data[i].rot_inv[8];

		// -- Set the lower corner (minimum) coordinates at the normalized orientation: +Y. The detector has thickness 0.
		detector_data[i].corner_min_rotated_to_Y.x = detector_data[i].corner_min_rotated_to_Y.x - 0.5 * detector_data[i].width_X;
		//  detector_data[i].corner_min_rotated_to_Y.y = detector_data[i].corner_min_rotated_to_Y.y;
		detector_data[i].corner_min_rotated_to_Y.z = detector_data[i].corner_min_rotated_to_Y.z - 0.5 * detector_data[i].height_Z;

		// *** Init the fan beam source model:

		rotZ = -rotZ; // The source rotation is the inverse of the detector.
		cos_rX = cos(rotX);
		cos_rZ = cos(rotZ);
		sin_rX = sin(rotX);
		sin_rZ = sin(rotZ);
		// --Rotation around X (alpha) and then around Z (phi): Rz*Rx (oposite of detector rotation)
		source_data[i].rot_fan[0] = cos_rZ;
		source_data[i].rot_fan[1] = -cos_rX * sin_rZ;
		source_data[i].rot_fan[2] = sin_rX * sin_rZ;
		source_data[i].rot_fan[3] = sin_rZ;
		source_data[i].rot_fan[4] = cos_rX * cos_rZ;
		source_data[i].rot_fan[5] = -sin_rX * cos_rZ;
		source_data[i].rot_fan[6] = 0.0f;
		source_data[i].rot_fan[7] = sin_rX;
		source_data[i].rot_fan[8] = cos_rX;
	}
	return true;
}

bool MCGPU::init_energy_spectrum(char *file_name_espc,
								 source_energy_struct *source_energy_data, 
								 float *mean_energy_spectrum)
{
	mexPrintf("    -- Reading the energy spectrum and initializing the Walker aliasing sampling algorithm.\n");
	char *new_line_ptr = NULL, new_line[250];
	float lower_energy_bin, prob;
	float prob_espc_bin[MAX_ENERGY_BINS]; // The input probabilities of each energy bin will be discarded after Walker is initialized

	// -- Read spectrum from file:
	FILE* file_ptr;
	fopen_s(&file_ptr, file_name_espc, "r");
	if (NULL == file_ptr)
	{
		mexPrintf("\n\n   !!init_energy_spectrum ERROR!! Error trying to read the energy spectrum input file \"%s\".\n\n", file_name_espc);
		return false;
	}

	int current_bin = -1;
	do
	{
		current_bin++; // Update bin counter

		if (current_bin >= MAX_ENERGY_BINS)
		{
			mexPrintf("\n !!init_energy_spectrum ERROR!!: too many energy bins in the input spectrum. Increase the value of MAX_ENERGY_BINS=%d.\n", MAX_ENERGY_BINS);
			mexPrintf("            A negative probability marks the end of the spectrum.\n\n");
			return false;
		}

		new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); // Read the following line of text skipping comments and extra spaces

		if (new_line_ptr == NULL)
		{
			mexPrintf("\n\n   !!init_energy_spectrum ERROR!! The input file for the x ray spectrum (%s) is not readable or incomplete (a negative probability marks the end of the spectrum).\n", file_name_espc);
			return false;
		}

		prob = -123456789.0f;

		sscanf_s(new_line, "%f %f", &lower_energy_bin, &prob);// Extract the lowest energy in the bin and the corresponding emission probability from the line read

		prob_espc_bin[current_bin] = prob;
		source_energy_data->espc[current_bin] = lower_energy_bin;

		if (prob == -123456789.0f)
		{
			mexPrintf("\n !!init_energy_spectrum ERROR!!: invalid energy bin number %d?\n\n", current_bin);
			return false;
		}
		else if (lower_energy_bin < source_energy_data->espc[max_value(current_bin - 1, 0)]) // (Avoid a negative index using the macro "max_value" defined in the header file)
		{
			mexPrintf("\n !!init_energy_spectrum ERROR!!: input energy bins with decreasing energy? espc(%d)=%f, espc(%d)=%f\n\n", current_bin - 1, source_energy_data->espc[max_value(current_bin - 1, 0)], current_bin, lower_energy_bin);
			return false;
		}

	} while (prob > -1.0e-11f); // A negative probability marks the end of the spectrum

	// Store the number of bins read from the input energy spectrum file:
	source_energy_data->num_bins_espc = current_bin;

	// Init the remaining bins (which will not be used) with the last energy read (will be assumed as the highest energy in the last bin) and 0 probability of emission.
	register int i;
	for (i = current_bin; i < MAX_ENERGY_BINS; i++)
	{
		source_energy_data->espc[i] = lower_energy_bin;
		prob_espc_bin[i] = 0.0f;
	}

	// Compute the mean energy in the spectrum, taking into account the energy and prob of each bin:
	float all_energy = 0.0f;
	float all_prob = 0.0f;
	for (i = 0; i < source_energy_data->num_bins_espc; i++)
	{
		all_energy += 0.5f * (source_energy_data->espc[i] + source_energy_data->espc[i + 1]) * prob_espc_bin[i];
		all_prob += prob_espc_bin[i];
	}
	*mean_energy_spectrum = all_energy / all_prob;

	// -- Init the Walker aliasing sampling method (as it is done in PENELOPE):
	IRND0(prob_espc_bin, source_energy_data->espc_cutoff, source_energy_data->espc_alias, source_energy_data->num_bins_espc); //!!Walker!! Calling PENELOPE's function to init the Walker method
	return true;
}

void MCGPU::update_seed_PRNG(int batch_number, unsigned long long total_histories,
					  int *seed)
{
	if (0 == batch_number)
		return;

	unsigned long long int leap = total_histories * (batch_number * LEAP_DISTANCE);
	int y = 1;
	int z = a1_RANECU;
	// -- Calculate the modulo power '(a^leap)MOD(m)' using a divide-and-conquer algorithm adapted to modulo arithmetic
	for (;;)
	{
		// (A2) Halve n, and store the integer part and the residue
		if (0 != (leap & 01)) // (bit-wise operation for MOD(leap,2), or leap%2 ==> proceed if leap is an odd number)  Equivalent: t=(short)(leap%2);
		{
			leap >>= 1;					 // Halve n moving the bits 1 position right. Equivalent to:  leap=(leap/2);
			y = abMODm(m1_RANECU, z, y); // (A3) Multiply y by z:  y = [z*y] MOD m
			if (0 == leap)
				break; // (A4) leap==0? ==> finish
		}
		else // (leap is even)
		{
			leap >>= 1; // Halve leap moving the bits 1 position right. Equivalent to:  leap=(leap/2);
		}
		z = abMODm(m1_RANECU, z, z); // (A5) Square z:  z = [z*z] MOD m
	}
	// AjMODm1 = y;                 // Exponentiation finished:  AjMODm = expMOD = y = a^j
	// -- Compute and display the seeds S(i+j), from the present seed S(i), using the previously calculated value of (a^j)MOD(m):
	//         S(i+j) = [(a**j MOD m)*S(i)] MOD m
	//         S_i = abMODm(m,S_i,AjMODm)
	*seed = abMODm(m1_RANECU, *seed, y);
}

void MCGPU::IRND0(float *W, float *F, short int *K, int N)
{
		register int I;

	//  ****  Renormalisation.
	double WS = 0.0;
	for (I = 0; I < N; I++)
	{
		if (W[I] < 0.0f)
		{
			mexPrintf("\n\n !!ERROR!! IRND0: Walker sampling initialization. Negative point probability? W(%d)=%f\n\n", I, W[I]);
			return;
		}
		WS = WS + W[I];
	}
	WS = ((double)N) / WS;

	for (I = 0; I < N; I++)
	{
		K[I] = I;
		F[I] = W[I] * WS;
	}

	if (N == 1)
		return;

	//  ****  Cutoff and alias values.
	float HLOW, HIGH;
	int ILOW, IHIGH, J;
	for (I = 0; I < N - 1; I++)
	{
		HLOW = 1.0f;
		HIGH = 1.0f;
		ILOW = -1;
		IHIGH = -1;
		for (J = 0; J < N; J++)
		{
			if (K[J] == J)
			{
				if (F[J] < HLOW)
				{
					HLOW = F[J];
					ILOW = J;
				}
				else if (F[J] > HIGH)
				{
					HIGH = F[J];
					IHIGH = J;
				}
			}
		}

		if ((ILOW == -1) || (IHIGH == -1))
			return;

		K[ILOW] = IHIGH;
		F[IHIGH] = HIGH + HLOW - 1.0f;
	}
	return;
}

bool MCGPU::_track_particles(dim3 block, dim3 thread, int num_p)
{
	return track_particles(block, thread, histories_per_thread, num_p, 
		seed_input, image_device, voxels_Edep_device, voxel_mat_dens_device, 
		mfp_Woodcock_table_device, mfp_table_a_device, mfp_table_b_device, 
		rayleigh_table_device, compton_table_device, detector_data_device, 
		source_data_device, materials_dose_device);
}

bool MCGPU::report_image_to_array(float* noScatter, float* compton, float* rayleigh, 
	float* multiscatter)
{
	const double SCALE = 1.0/SCALE_eV;    // conversion to eV using the inverse of the constant used in the "tally_image" kernel function (defined in the header file)
	const double NORM = SCALE * detector_data[0].inv_pixel_size_X * detector_data[0].inv_pixel_size_Z / ((double)total_histories);  // ==> [eV/cm^2 per history]
	int pixels_per_image = detector_data[0].total_num_pixels;
	for (int i = 0; i < pixels_per_image; ++i) {
		noScatter[i] = (float)( NORM * (double)(image[i]) );
		compton[i] = (float)( NORM * (double)(image[i + pixels_per_image]));
		rayleigh[i] = (float)( NORM * (double)(image[i + 2*pixels_per_image]));
		multiscatter[i] = (float)( NORM * (double)(image[i + 3*pixels_per_image]));
	}
	return true;
}

void MCGPU::_print_init()
{
    mexPrintf("\n\n     *****************************************************************************\n");
    mexPrintf("     ***         MC-GPU, version 1.3 (http://code.google.com/p/mcgpu/)         ***\n");
    mexPrintf("     ***                                                                       ***\n");
    mexPrintf("     ***  A. Badal and A. Badano, \"Accelerating Monte Carlo simulations of     *** \n");
    mexPrintf("     ***  photon transport in a voxelized geometry using a massively parallel  *** \n");
    mexPrintf("     ***  Graphics Processing Unit\", Medical Physics 36, pp. 4878–4880 (2009)  ***\n");
    mexPrintf("     ***                                                                       ***\n");
    mexPrintf("     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
    mexPrintf("     *****************************************************************************\n\n");
    auto current_time = time(nullptr);
    mexPrintf("****** Code execution started on: %s\n\n", ctime(&current_time));
    

	// The "MASTER_THREAD" macro prints the messages just once when using MPI threads (it has no effect if MPI is not used):  MASTER_THREAD == "if(0==myID)"
	mexPrintf("\n             *** CUDA SIMULATION IN THE GPU ***\n");

	mexPrintf("\n    -- INITIALIZATION phase:\n");
	 // Clear the screen output buffer for the master thread

    return;
}

void MCGPU::_print_read()
{
    // *** Output some of the data read to make sure everything was correctly read:
    if (total_histories < (unsigned long long int)(100000))
        mexPrintf("                       simulation time = %lld s\n", total_histories);
    else
        mexPrintf("              x-ray tracks to simulate = %lld\n", total_histories);
    mexPrintf("                   initial random seed = %d\n", seed_input);
    mexPrintf("      azimuthal (phi), polar apertures = %.6f , %.6f degrees\n", ((double)source_data[0].D_phi) * RAD2DEG, 2.0 * (90.0 - acos(((double)source_data[0].cos_theta_low)) * RAD2DEG));
    mexPrintf("                   focal spot position = (%f, %f, %f)\n", source_data[0].position.x, source_data[0].position.y, source_data[0].position.z);
    mexPrintf("                      source direction = (%f, %f, %f)\n", source_data[0].direction.x, source_data[0].direction.y, source_data[0].direction.z);
    mexPrintf("                  initial angle from X = %lf\n", initial_angle * RAD2DEG);
    mexPrintf("              source-detector distance = %f cm\n", detector_data[0].sdd);
    mexPrintf("                       detector center = (%f, %f, %f)\n", (source_data[0].position.x + source_data[0].direction.x * detector_data[0].sdd), // Center of the detector straight ahead of the focal spot.
           (source_data[0].position.y + source_data[0].direction.y * detector_data[0].sdd),
           (source_data[0].position.z + source_data[0].direction.z * detector_data[0].sdd));
    mexPrintf("           detector low corner (at +Y) = (%f, %f, %f)\n", detector_data[0].corner_min_rotated_to_Y.x, detector_data[0].corner_min_rotated_to_Y.y, detector_data[0].corner_min_rotated_to_Y.z);
    mexPrintf("                number of pixels image = %dx%d = %d\n", detector_data[0].num_pixels.x, detector_data[0].num_pixels.y, detector_data[0].total_num_pixels);
    mexPrintf("                            pixel size = %.3fx%.3f cm\n", 1.0f / detector_data[0].inv_pixel_size_X, 1.0f / detector_data[0].inv_pixel_size_Z);
    mexPrintf("                 number of projections = %d\n", num_projections);
    if (num_projections != 1)
    {
        mexPrintf("         source-rotation axis-distance = %lf cm\n", SRotAxisD);
        mexPrintf("             angle between projections = %lf\n", D_angle * RAD2DEG);
        mexPrintf("            angular region of interest = [%lf,%lf] degrees\n", angularROI_0 * RAD2DEG, angularROI_1 * RAD2DEG);
        mexPrintf("   vertical translation per projection = %lf cm\n", vertical_translation_per_projection);
    }
    mexPrintf("                      Input voxel file = %s\n", file_name_voxels);
    mexPrintf("                     Output image file = %s\n", file_name_output);
    if (output_proj_type == 1)
        mexPrintf("                     Output image type = Raw\n");
    else if (output_proj_type == 2)
        mexPrintf("                     Output image type = Ascii\n");
    else if (output_proj_type == 3)
        mexPrintf("                     Output image type = Raw and Ascii\n");

    if (dose_ROI_x_max > -1)
    {
        mexPrintf("                      Output dose file = %s\n", file_dose_output);
        mexPrintf("         Input region of interest dose = X[%d,%d], Y[%d,%d], Z[%d,%d]\n", dose_ROI_x_min + 1, dose_ROI_x_max + 1, dose_ROI_y_min + 1, dose_ROI_y_max + 1, dose_ROI_z_min + 1, dose_ROI_z_max + 1); // Show ROI with index=1 for the first voxel instead of 0.
    }

    mexPrintf("\n                  Energy spectrum file = %s\n", file_name_espc);
    mexPrintf("            number of energy bins read = %d\n", source_energy_data.num_bins_espc);
    mexPrintf("             minimum, maximum energies = %.3f, %.3f keV\n", 0.001f * source_energy_data.espc[0], 0.001f * source_energy_data.espc[source_energy_data.num_bins_espc]);
    mexPrintf("                  mean energy spectrum = %.3f keV\n\n", 0.001f * mean_energy_spectrum);

    
}

void MCGPU::_print_allocated()
{
    mexPrintf("       Total CPU memory allocated for voxels vector and data structures = %f Mbytes\n", (voxel_mat_dens_bytes + image_bytes + sizeof(voxel_struct) + sizeof(source_struct) + sizeof(detector_struct) + sizeof(linear_interp) + 2 * mfp_table_bytes + sizeof(rayleigh_struct) + sizeof(compton_struct)) / (1024.f * 1024.f));
	
    return;
}

bool MCGPU::_check_consistent()
{
    // -- Check that the input material tables and the x-ray source are consistent:
	if ((source_energy_data.espc[0] < mfp_table_data.e0) || (source_energy_data.espc[source_energy_data.num_bins_espc] > (mfp_table_data.e0 + (mfp_table_data.num_values - 1) / mfp_table_data.ide)))
	{
		mexPrintf("\n\n\n !!ERROR!! The input x-ray source energy spectrum minimum (%.3f eV) and maximum (%.3f eV) energy values\n", source_energy_data.espc[0], source_energy_data.espc[source_energy_data.num_bins_espc]);
		mexPrintf("           are outside the tabulated energy interval for the material properties tables (from %.3f to %.3f eV)!!\n", mfp_table_data.e0, (mfp_table_data.e0 + (mfp_table_data.num_values - 1) / mfp_table_data.ide));
		mexPrintf("           Please, modify the input energy spectra to fit the tabulated limits or create new tables.\n\n");
		return false;
	}
    return true;
}

void MCGPU::_print_mc_start(time_t clock_start)
{
    mexPrintf("\n    -- INITIALIZATION finished: elapsed time = %.3f s. \n\n", ((double)(clock() - clock_start)) / CLOCKS_PER_SEC);
    time_t current_time = time(NULL);
    mexPrintf("\n\n    -- MONTE CARLO LOOP phase. Time: %s\n\n", ctime(&current_time));
    return;
}

bool MCGPU::_check_angle(double current_angle, int num_p)
{
    if ((current_angle < angularROI_0) || (current_angle > angularROI_1))
    {
        mexPrintf("         << Skipping projection #%d of %d >> Angle %f degrees: outside angular region of interest.\n", num_p + 1, num_projections, current_angle * RAD2DEG);
        return false; // Cycle loop: do not simulate this projection!
    }

    if (num_projections != 1)
        mexPrintf("\n\n\n   << Simulating Projection %d of %d >> Angle: %lf degrees.\n\n\n", num_p + 1, num_projections, current_angle * RAD2DEG);
    return true;
}

void MCGPU::_print_finish(clock_t clock_start_beginning, double time_total_MC_simulation)
{
    mexPrintf("\n\n\n    -- SIMULATION FINISHED!\n");

    double time_total_MC_init_report = ((double)(clock() - clock_start_beginning)) / CLOCKS_PER_SEC;

    // -- Report total performance:
    mexPrintf("\n\n       ****** TOTAL SIMULATION PERFORMANCE (including initialization and reporting) ******\n\n");
    mexPrintf("          >>> Execution time including initialization, transport and report: %.3f s.\n", time_total_MC_init_report);
    mexPrintf("          >>> Time spent in the Monte Carlo transport only: %.3f s.\n", time_total_MC_simulation);
    mexPrintf("          >>> Time spent in initialization, reporting and clean up: %.3f s.\n\n", (time_total_MC_init_report - time_total_MC_simulation));
    mexPrintf("          >>> Total number of simulated x rays:  %lld\n", total_histories * ((unsigned long long int)num_projections));
    if (time_total_MC_init_report > 0.000001)
        mexPrintf("          >>> Total speed (using %d thread, including initialization time) [x-rays/s]:  %.2f\n\n", 1, (double)(total_histories * ((unsigned long long int)num_projections)) / time_total_MC_init_report);

    time_t current_time = time(NULL); // Get current time (in seconds)

    mexPrintf("\n****** Code execution finished on: %s\n\n", ctime(&current_time));
    return;
}

void MCGPU::_check_block(int &total_threads, int &total_threads_blocks)
{
	// -- Compute the number of CUDA blocks to simulate, rounding up and making sure it is below the limit of 65535 blocks.
	//    The total number of particles simulated will be increased to the nearest multiple "histories_per_thread".
	total_threads = (int)(((double)total_histories) / ((double)histories_per_thread) + 0.9990);		  // Divide the histories among GPU threads, rounding up
	total_threads_blocks = (int)(((double)total_threads) / ((double)num_threads_per_block) + 0.9990); // Divide the GPU threads among CUDA blocks, rounding up
	if (total_threads_blocks > 65535)
	{
		mexPrintf("\n          WARNING: %d hist per thread would produce %d CUDA blocks, more than the maximum value of 65535.", histories_per_thread, total_threads_blocks);
		total_threads_blocks = 65000; // Increase the histories per thread to have exactly 65000 blocks.
		histories_per_thread = (int)(((double)total_histories) / ((double)(total_threads_blocks * num_threads_per_block)) + 0.9990);
		mexPrintf(" Increasing to %d hist to run exactly %d blocks in the GPU.\n", histories_per_thread, total_threads_blocks);
	}
	else if (total_threads_blocks < 1)
	{
		total_threads_blocks = 1; // Make sure we have at least 1 block to run
	}

	total_histories = ((unsigned long long int)(total_threads_blocks * num_threads_per_block)) * histories_per_thread; // Total histories will be equal or higher than the input value due to the rounding up in the division of the histories

	mexPrintf("\n        ==> CUDA: Executing %d blocks of %d threads, with %d histories in each thread: %lld histories in total (random seed: %d).\n", total_threads_blocks, num_threads_per_block, histories_per_thread, total_histories, seed_input);
	
	return;
}