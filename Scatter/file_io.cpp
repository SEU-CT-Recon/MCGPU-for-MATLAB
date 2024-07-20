#include "file_io.h"
#include <fstream>
#include <mex.h>

////////////////////////////////////////////////////////////////////////////////
//! Read a line of text and trim initial blancks and trailing comments (#).
//!
//!       @param[in] num   Characters to read
//!       @param[in] file_ptr   Pointer to the input file stream
//!       @param[out] trimmed_line   Trimmed line from input file, skipping empty lines and comments
////////////////////////////////////////////////////////////////////////////////
char *fgets_trimmed(char *trimmed_line, int num, FILE *file_ptr)
{
    char new_line[250];
    char *new_line_ptr = NULL;
    int a = 0, b = 0;
    trimmed_line[0] = '\0'; //  Init with a mark that means no file input

    do
    {
        a = 0;
        b = 0;
        new_line_ptr = fgets(new_line, num, file_ptr); // Read new line
        if (new_line_ptr != NULL)
        {
            // Discard initial blanks:
            while (' ' == new_line[a])
            {
                a++;
            }
            // Read file until a comment symbol (#) or end-of-line are found:
            while (('\n' != new_line[a]) && ('#' != new_line[a]))
            {
                trimmed_line[b] = new_line[a];
                b++;
                a++;
            }
        }
    } while (new_line_ptr != NULL && '\0' == trimmed_line[0]); // Keep reading lines until end-of-file or a line that is not empty or only comment is found

    trimmed_line[b] = '\0'; // Terminate output string
    return new_line_ptr;
}

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
bool load_material(char file_name_materials[MAX_MATERIALS][250], float *density_max, float *density_nominal, struct linear_interp *mfp_table_data, float2 **mfp_Woodcock_table_ptr, int *mfp_Woodcock_table_bytes, float3 **mfp_table_a_ptr, float3 **mfp_table_b_ptr, int *mfp_table_bytes, struct rayleigh_struct *rayleigh_table_ptr, struct compton_struct *compton_table_ptr)
{
  char new_line[250];
  char *new_line_ptr = NULL;
  int mat, i, bin, input_num_values = 0, input_rayleigh_values = 0, input_num_shells = 0;
  double delta_e=-99999.0;

  // -- Init the number of shells to 0 for all materials
  for (mat=0; mat<MAX_MATERIALS; mat++)
    compton_table_ptr->noscco[mat] = 0;
    

  // --Read the material data files:
  mexPrintf("\n    -- Reading the material data files (MAX_MATERIALS=%d):\n", MAX_MATERIALS);
  for (mat=0; mat<MAX_MATERIALS; mat++)
  {
    if ((file_name_materials[mat][0]=='\0') || (file_name_materials[mat][0]=='\n'))  //  Empty file name
       continue;   // Re-start loop for next material

    mexPrintf("         Mat %d: File \'%s\'\n", mat+1, file_name_materials[mat]);
//     mexPrintf("    -- Reading material file #%d: \'%s\'\n", mat, file_name_materials[mat]);

    gzFile file_ptr = gzopen(file_name_materials[mat], "rb");    // !!zlib!!  
    if (file_ptr==NULL)
    {
      mexPrintf("\n\n   !!fopen ERROR!! File %d \'%s\' does not exist!!\n", mat, file_name_materials[mat]);
      return false;
    }
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   // Read full line (max. 250 characters).   //  !!zlib!!
      if (new_line_ptr==NULL)
      {
        mexPrintf("\n\n   !!Reading ERROR!! File is not readable or does not contain the string \'[NOMINAL DENSITY\'!!\n");
        return false;
      }
    }
    while(strstr(new_line,"[NOMINAL DENSITY")==NULL);   // Skip rest of the header

    // Read the material nominal density:
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    sscanf(new_line, "# %f", &density_nominal[mat]);
    
    if (density_max[mat]>0)    //  Material found in the voxels
    {
      mexPrintf("                Nominal density = %f g/cm^3; Max density in voxels = %f g/cm^3\n", density_nominal[mat], density_max[mat]);
    }
    else                       //  Material NOT found in the voxels
    {
      mexPrintf("                This material is not used in any voxel.\n");
      
      // Do not lose time reading the data for materials not found in the voxels, except for the first one (needed to determine the size of the input data).      
      if (0 == mat)
        density_max[mat] = 0.01f*density_nominal[mat];   // Assign a small but positive density; this material will not be used anyway.
      else
        continue;     //  Move on to next material          
    }
      

    // --For the first material, set the number of energy values and allocate table arrays:
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    sscanf(new_line, "# %d", &input_num_values);
    if (0==mat)
    {
      mfp_table_data->num_values = input_num_values;
      mexPrintf("                Number of energy values in the mean free path database: %d.\n", input_num_values);

      // Allocate memory for the linear interpolation arrays:
      *mfp_Woodcock_table_bytes = sizeof(float2)*input_num_values;
      *mfp_Woodcock_table_ptr   = (float2*) malloc(*mfp_Woodcock_table_bytes);  // Allocate space for the 2 parameter table
      *mfp_table_bytes = sizeof(float3)*input_num_values*MAX_MATERIALS;
      *mfp_table_a_ptr = (float3*) malloc(*mfp_table_bytes);  // Allocate space for the 4 MFP tables
      *mfp_table_b_ptr = (float3*) malloc(*mfp_table_bytes);
      *mfp_table_bytes = sizeof(float3)*input_num_values*MAX_MATERIALS;

      if (input_num_values>MAX_ENERGYBINS_RAYLEIGH)
      {
        mexPrintf("\n\n   !!load_material ERROR!! Too many energy bins (Input bins=%d): increase parameter MAX_ENERGYBINS_RAYLEIGH=%d!!\n\n", input_num_values, MAX_ENERGYBINS_RAYLEIGH);
        return false;
      }
      
      if ((NULL==*mfp_Woodcock_table_ptr)||(NULL==*mfp_table_a_ptr)||(NULL==*mfp_table_b_ptr))
      {
        mexPrintf("\n\n   !!malloc ERROR!! Not enough memory to allocate the linear interpolation data: %d bytes!!\n\n", (*mfp_Woodcock_table_bytes+2*(*mfp_table_bytes)));
        return false;
      }
      else
      {
        mexPrintf("                Linear interpolation data correctly allocated (%f Mbytes)\n", (*mfp_Woodcock_table_bytes+2*(*mfp_table_bytes))/(1024.f*1024.f));
      }
      for (i=0; i<input_num_values; i++)
      {
        (*mfp_Woodcock_table_ptr)[i].x = 99999999.99f;    // Init this array with a huge MFP, the minimum values are calculated below
      }
    }
    else   // Materials after first
    {
      if (input_num_values != mfp_table_data->num_values)
      {
        mexPrintf("\n\n   !!load_material ERROR!! Incorrect number of energy values given in material \'%s\': input=%d, expected=%d\n",file_name_materials[mat], input_num_values, mfp_table_data->num_values);
        return false;
      }
    }

    // -- Read the mean free paths (and Rayleigh cumulative prob):
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    double d_energy, d_rayleigh, d_compton, d_photelectric, d_total_mfp, d_pmax, e_last=-1.0;
    
    for (i=0; i<input_num_values; i++)
    {

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line,"  %le  %le  %le  %le  %le  %le", &d_energy, &d_rayleigh, &d_compton, &d_photelectric, &d_total_mfp, &d_pmax);

      // Find and store the minimum total MFP at the current energy, for every material's maximum density:
      float temp_mfp = d_total_mfp*(density_nominal[mat])/(density_max[mat]);
      if (temp_mfp < (*mfp_Woodcock_table_ptr)[i].x)
        (*mfp_Woodcock_table_ptr)[i].x = temp_mfp;       // Store minimum total mfp [cm]

      // Store the inverse MFP data points with [num_values rows]*[MAX_MATERIALS columns]
      // Scaling the table to the nominal density so that I can re-scale in the kernel to the actual local density:
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].x = 1.0/(d_total_mfp*density_nominal[mat]);   // inverse TOTAL mfp * nominal density
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].y = 1.0/(d_compton  *density_nominal[mat]);   // inverse Compton mfp * nominal density
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].z = 1.0/(d_rayleigh *density_nominal[mat]);   // inverse Rayleigh mfp * nominal density

      rayleigh_table_ptr->pmax[i*(MAX_MATERIALS)+mat] = d_pmax;    // Store the maximum cumulative probability of atomic form factor F^2 for

      if (0==i && 0==mat)
      {
        mfp_table_data->e0  = d_energy;   // Store the first energy of the first material
      }

      if (0==i)
      {
        if (fabs(d_energy-mfp_table_data->e0)>1.0e-9)
        {
          mexPrintf("\n\n   !!load_material ERROR!! Incorrect first energy value given in material \'%s\': input=%f, expected=%f\n", file_name_materials[mat], d_energy, mfp_table_data->e0);
          return false;
        }
      }
      else if (1==i)
      {
        delta_e = d_energy-e_last;
      }
      else if (i>1)
      {
        if (((fabs((d_energy-e_last)-delta_e))/delta_e)>0.001)  // Tolerate up to a 0.1% relative variation in the delta e (for each bin) to account for possible precission errors reading the energy values
        {
          mexPrintf("  !!ERROR reading material data!! The energy step between mean free path values is not constant!!\n      (maybe not enough decimals given for the energy values)\n      #value = %d, First delta: %f , New delta: %f, Energy: %f ; Rel.Dif=%f\n", i, delta_e, (d_energy-e_last), d_energy,((fabs((d_energy-e_last)-delta_e))/delta_e));
          return false;
        }
      }
      e_last = d_energy;
    }
    
    if (0==mat) mexPrintf("                Lowest energy first bin = %f eV, last bin = %f eV; bin width = %f eV\n", (mfp_table_data->e0), e_last, delta_e);

    // -- Store the inverse of delta energy:
    mfp_table_data->ide = 1.0f/delta_e;

    // -- Store MFP data slope 'b' (.y for Woodcock):
    for (i=0; i<(input_num_values-1); i++)
    {
      bin = i*MAX_MATERIALS+mat;                   // Set current bin, skipping MAX_MATERIALS columns
      (*mfp_table_b_ptr)[bin].x = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].x - (*mfp_table_a_ptr)[bin].x) / delta_e;
      (*mfp_table_b_ptr)[bin].y = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].y - (*mfp_table_a_ptr)[bin].y) / delta_e;
      (*mfp_table_b_ptr)[bin].z = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].z - (*mfp_table_a_ptr)[bin].z) / delta_e;
    }
    // After maximum energy (last bin), assume constant slope:
    (*mfp_table_b_ptr)[(input_num_values-1)*MAX_MATERIALS+mat] = (*mfp_table_b_ptr)[(input_num_values-2)*MAX_MATERIALS+mat];

    // -- Rescale the 'a' parameter (.x for Woodcock) as if the bin started at energy = 0: we will not have to rescale to the bin minimum energy every time
    for (i=0; i<input_num_values; i++)
    {
      d_energy = mfp_table_data->e0 + i*delta_e;   // Set current bin lowest energy value
      bin = i*MAX_MATERIALS+mat;                   // Set current bin, skipping MAX_MATERIALS columns
      (*mfp_table_a_ptr)[bin].x = (*mfp_table_a_ptr)[bin].x - d_energy*(*mfp_table_b_ptr)[bin].x;
      (*mfp_table_a_ptr)[bin].y = (*mfp_table_a_ptr)[bin].y - d_energy*(*mfp_table_b_ptr)[bin].y;
      (*mfp_table_a_ptr)[bin].z = (*mfp_table_a_ptr)[bin].z - d_energy*(*mfp_table_b_ptr)[bin].z;
    }

    // -- Reading data for RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database):
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      if (gzeof(file_ptr)!=0)                           //  !!zlib!!
      {
        mexPrintf("\n\n   !!End-of-file ERROR!! Rayleigh data not found: \"#[DATA VALUES...\" in file \'%s\'. Last line read: %s\n\n", file_name_materials[mat], new_line);
        return false;
      }
    }
    while(strstr(new_line,"[DATA VALUES")==NULL);   // Skip all lines until this text is found
      
    new_line_ptr = gzgets(file_ptr, new_line, 250);   // Read the number of data points in Rayleigh     //  !!zlib!! 
    sscanf(new_line, "# %d", &input_rayleigh_values);
        
    if (input_rayleigh_values != NP_RAYLEIGH)
    {
      mexPrintf("\n\n   !!ERROR!! The number of values for Rayleigh sampling is different than the allocated space: input=%d, NP_RAYLEIGH=%d. File=\'%s\'\n", input_rayleigh_values, NP_RAYLEIGH, file_name_materials[mat]);
      return false;
    }
    new_line_ptr = gzgets(file_ptr, new_line, 250);    // Comment line:  #[SAMPLING DATA FROM COMMON/CGRA/: X, P, A, B, ITL, ITU]     //  !!zlib!!
    for (i=0; i<input_rayleigh_values; i++)
    {
      int itlco_tmp, ituco_tmp;
      bin = NP_RAYLEIGH*mat + i;

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line,"  %e  %e  %e  %e  %d  %d", &(rayleigh_table_ptr->xco[bin]), &(rayleigh_table_ptr->pco[bin]),
                                                  &(rayleigh_table_ptr->aco[bin]), &(rayleigh_table_ptr->bco[bin]),
                                                  &itlco_tmp, &ituco_tmp);

      rayleigh_table_ptr->itlco[bin] = (unsigned char) itlco_tmp;
      rayleigh_table_ptr->ituco[bin] = (unsigned char) ituco_tmp;
                                                  
    }
    //  mexPrintf("    -- Rayleigh sampling data read. Input values = %d\n",input_rayleigh_values);

    // -- Reading COMPTON INTERACTIONS data (relativistic impulse model with approximated one-electron analytical profiles):
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      if (gzeof(file_ptr)!=0)                           //  !!zlib!!
      {
        mexPrintf("\n\n   !!End-of-file ERROR!! Compton data not found: \"[NUMBER OF SHELLS]\" in file \'%s\'. Last line read: %s\n\n", file_name_materials[mat], new_line);
        return false;
      }
    }
    while(strstr(new_line,"[NUMBER OF SHELLS")==NULL);   // Skip all lines until this text is found
    new_line_ptr = gzgets(file_ptr, new_line, 250);
    sscanf(new_line, "# %d", &input_num_shells);      // Read the NUMBER OF SHELLS
    if (input_num_shells>MAX_SHELLS)
    {
      mexPrintf("\n\n   !!ERROR!! Too many shells for Compton interactions in file \'%s\': input=%d, MAX_SHELLS=%d\n", file_name_materials[mat], input_num_shells, MAX_SHELLS);
      return false;
    }
    compton_table_ptr->noscco[mat] = input_num_shells;   // Store number of shells for this material in structure
    new_line_ptr = gzgets(file_ptr, new_line, 250);      // Comment line:  #[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]
    int kzco_dummy, ksco_dummy;
    for (i=0; i<input_num_shells; i++)
    {

      bin = mat + i*MAX_MATERIALS;

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line," %e  %e  %e  %d  %d", &(compton_table_ptr->fco[bin]), &(compton_table_ptr->uico[bin]),
                                              &(compton_table_ptr->fj0[bin]), &kzco_dummy, &ksco_dummy);
    }
  
    gzclose(file_ptr);    // Material data read. Close the current material input file.           //  !!zlib!!
    
  }  // ["for" loop: continue with next material]


  // -- Store Woodcock MFP slope in component '.y':
  for (i=0; i<(mfp_table_data->num_values-1); i++)
    (*mfp_Woodcock_table_ptr)[i].y = ((*mfp_Woodcock_table_ptr)[i+1].x - (*mfp_Woodcock_table_ptr)[i].x)/delta_e;

  // -- Rescale the first parameter in component .x for Woodcock
  for (i=0; i<mfp_table_data->num_values; i++)
  {
    (*mfp_Woodcock_table_ptr)[i].x = (*mfp_Woodcock_table_ptr)[i].x - (mfp_table_data->e0 + i*delta_e)*(*mfp_Woodcock_table_ptr)[i].y;
  }
  return true;
}

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
int report_image(char* file_name_output, int r_output_proj_type, struct detector_struct* detector_data, struct source_struct* source_data, float mean_energy_spectrum, unsigned long long int* image, double time_elapsed, unsigned long long int total_histories, int current_projection, int num_projections, double D_angle, double initial_angle)
{
  
  //  -Find current angle
  double current_angle = initial_angle+current_projection*D_angle;

  // -- Report data:
  mexPrintf("\n\n          *** IMAGE TALLY PERFORMANCE REPORT ***\n");
  
  if(num_projections!=1)   // Output the projection angle when simulating a CT:
  {
    mexPrintf("              CT projection %d of %d: angle from X axis = %lf \n", current_projection+1, num_projections, current_angle*RAD2DEG);
  }
  
  mexPrintf("              Simulated x rays:    %lld\n", total_histories);
  mexPrintf("              Simulation time [s]: %.2f\n", time_elapsed);
  if (time_elapsed>0.000001)
    mexPrintf("              Speed [x-rays/s]:    %.2f\n\n", ((double)total_histories)/time_elapsed);

  const double SCALE = 1.0/SCALE_eV;    // conversion to eV using the inverse of the constant used in the "tally_image" kernel function (defined in the header file)
  const double NORM = SCALE * detector_data[0].inv_pixel_size_X * detector_data[0].inv_pixel_size_Z / ((double)total_histories);  // ==> [eV/cm^2 per history]
  int pixels_per_image = (detector_data[0].num_pixels.x*detector_data[0].num_pixels.y);

  mexPrintf("output_proj_type: %d\n",r_output_proj_type);
  mexPrintf("%s\n",file_name_output);

  // -- ASCII output: 
  if(r_output_proj_type & 0x0010)
  {
    double energy_noScatter, energy_compton, energy_rayleigh, energy_multiscatter;
    double energy_integral = 0.0;   // Integrate (add) the energy in the image pixels [meV]
    double maximum_energy_pixel = -100.0;  // Find maximum pixel signal
    int maximum_energy_pixel_x=0, maximum_energy_pixel_y=0, maximum_energy_pixel_number=0;  

    FILE* file_ptr = fopen(file_name_output, "w");
  
    if (file_ptr==NULL)
    {
      mexPrintf("\n\n   !!fopen ERROR report_image!! File %s can not be opened!!\n", file_name_output);
      exit(-3);
    }
    
    fprintf(file_ptr, "# \n");
    fprintf(file_ptr, "#     *****************************************************************************\n");
    fprintf(file_ptr, "#     ***         MC-GPU, version 1.3 (http://code.google.com/p/mcgpu/)         ***\n");
    fprintf(file_ptr, "#     ***                                                                       ***\n");
    fprintf(file_ptr, "#     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
    fprintf(file_ptr, "#     *****************************************************************************\n");
    fprintf(file_ptr, "# \n"); 
    fprintf(file_ptr, "#  *** SIMULATION IN THE GPU USING CUDA ***\n");
    fprintf(file_ptr, "#\n");
    fprintf(file_ptr, "#  Image created counting the energy arriving at each pixel: ideal energy integrating detector.\n");
    fprintf(file_ptr, "#  Pixel value units: eV/cm^2 per history (energy fluence).\n");
  
  
    if(num_projections!=1)   // Output the projection angle when simulating a CT:
    {
      fprintf(file_ptr, "#  CT projection %d of %d: angle from X axis = %lf \n", current_projection+1, num_projections, current_angle*RAD2DEG);
    }  
  
    fprintf(file_ptr, "#  Focal spot position = (%.8f,%.8f,%.8f), cone beam direction = (%.8f,%.8f,%.8f)\n", source_data[current_projection].position.x, source_data[current_projection].position.y, source_data[current_projection].position.z, source_data[current_projection].direction.x, source_data[current_projection].direction.y, source_data[current_projection].direction.z);
  
    fprintf(file_ptr, "#  Pixel size:  %lf x %lf = %lf cm^2\n", 1.0/(double)(detector_data[0].inv_pixel_size_X), 1.0/(double)(detector_data[0].inv_pixel_size_Z), 1.0/(double)(detector_data[0].inv_pixel_size_X*detector_data[0].inv_pixel_size_Z));
    
    fprintf(file_ptr, "#  Number of pixels in X and Z:  %d  %d\n", detector_data[0].num_pixels.x, detector_data[0].num_pixels.y);
    fprintf(file_ptr, "#  (X rows given first, a blank line separates the different Z values)\n");
    fprintf(file_ptr, "# \n");
    fprintf(file_ptr, "#  [NON-SCATTERED] [COMPTON] [RAYLEIGH] [MULTIPLE-SCATTING]\n");
    fprintf(file_ptr, "# ==========================================================\n");

    int i, j;
    int  pixel=0;
    double noScatter_integral = 0.0;
    double compton_integral = 0.0;
    double rayleigh_integral = 0.0;
    double multiscatter_integral = 0.0;
    for(j=0; j<detector_data[0].num_pixels.y; j++)
    {
      for(i=0; i<detector_data[0].num_pixels.x; i++)
      {
        energy_noScatter    = (double)(image[pixel]);
        energy_compton      = (double)(image[pixel +   pixels_per_image]);
        energy_rayleigh     = (double)(image[pixel + 2*pixels_per_image]);
        energy_multiscatter = (double)(image[pixel + 3*pixels_per_image]);
  
        // -- Write the results in an external file; the image corresponding to all particles not written: it has to be infered adding all images
        fprintf(file_ptr, "%.8lf %.8lf %.8lf %.8lf\n", NORM*energy_noScatter, NORM*energy_compton, NORM*energy_rayleigh, NORM*energy_multiscatter);
        
        register double total_energy_pixel = energy_noScatter + energy_compton + energy_rayleigh + energy_multiscatter;   // Find and report the pixel with maximum signal
        if (total_energy_pixel>maximum_energy_pixel)
        {
          maximum_energy_pixel = total_energy_pixel;
          maximum_energy_pixel_x = i;
          maximum_energy_pixel_y = j;
          maximum_energy_pixel_number = pixel;
        }            
        energy_integral += total_energy_pixel;   // Count total energy in the whole image      
        noScatter_integral += energy_noScatter;
        compton_integral += energy_compton;
        rayleigh_integral += energy_rayleigh;
        multiscatter_integral += energy_multiscatter;
        
        pixel++;   // Move to next pixel
      }
      fprintf(file_ptr, "\n");     // Separate rows with an empty line for visualization with gnuplot.
    }

    fprintf(file_ptr, "#   *** Simulation REPORT: ***\n");
    fprintf(file_ptr, "#       Fraction of energy detected (over the mean energy of the spectrum): %.3lf%%\n", 100.0*SCALE*(energy_integral/(double)(total_histories))/(double)(mean_energy_spectrum));
    fprintf(file_ptr, "#       Maximum energy detected in pixel %i: (x,y)=(%i,%i) -> pixel value = %lf eV/cm^2\n", maximum_energy_pixel_number, maximum_energy_pixel_x, maximum_energy_pixel_y, NORM*maximum_energy_pixel);
    fprintf(file_ptr, "#       Simulated x rays:    %lld\n", total_histories);
    fprintf(file_ptr, "#       Simulation time [s]: %.2f\n", time_elapsed);
    if (time_elapsed>0.000001)
      fprintf(file_ptr, "#       Speed [x-rays/sec]:  %.2f\n\n", ((double)total_histories)/time_elapsed);
     
    fprintf(file_ptr, "#        Scattering Ratio :\n");
    fprintf(file_ptr, "#        Compton:      %.3lf%%\n", 100.0*compton_integral/energy_integral);
    fprintf(file_ptr, "#        Rayleigh:     %.3lf%%\n", 100.0*rayleigh_integral/energy_integral);
    fprintf(file_ptr, "#        Multiscatter: %.3lf%%\n", 100.0*multiscatter_integral/energy_integral);
    fprintf(file_ptr, "#        Summary:      %.3lf%%\n", 100.0*(1-noScatter_integral/energy_integral));

    fclose(file_ptr);  // Close output file and flush stream

    mexPrintf("              Fraction of initial energy arriving at the detector (over the mean energy of the spectrum):  %.3lf%%\n", 100.0*SCALE*(energy_integral/(double)(total_histories))/(double)(mean_energy_spectrum));
    mexPrintf("              Maximum energy detected in pixel %i: (x,y)=(%i,%i). Maximum pixel value = %lf eV/cm^2\n\n", maximum_energy_pixel_number, maximum_energy_pixel_x, maximum_energy_pixel_y, NORM*maximum_energy_pixel);  
    fflush(stdout);
  }
  
  // -- Binary output: 
  if(r_output_proj_type & 0x0001)
  {
    float energy_float;
    char file_binary[250];
    strncpy (file_binary, file_name_output, 250);
    strcat(file_binary,".raw");                       // !!BINARY!! 
    std::fstream file_binary_ptr(file_binary, std::ios::out | std::ios::binary);
    if (!file_binary_ptr.is_open())
    {
      mexPrintf("\n\n   !!fopen ERROR report_image!! Binary file %s can not be opened for writing!!\n", file_binary);
      exit(-3);
    }

    int i;

    double noScatter_integral = 0.0;
    double compton_integral = 0.0;
    double rayleigh_integral = 0.0;
    double multiscatter_integral = 0.0;
    float* tmp = new float[pixels_per_image * 4+4];
    memset(tmp, 0, pixels_per_image * 4 * sizeof(float));

    for(i=0; i<pixels_per_image; i++)
    {
      energy_float = (float)( NORM * (double)(image[i]) );  // Non-scattered image
      noScatter_integral += energy_float;
      tmp[i] = energy_float;      
    }
    for(i=0; i<pixels_per_image; i++)
    {

      energy_float = (float)( NORM * (double)(image[i + pixels_per_image]) );  // Compton image
      compton_integral += energy_float;
      tmp[i + pixels_per_image] = energy_float;
    }
    for(i=0; i<pixels_per_image; i++)
    {
      energy_float = (float)( NORM * (double)(image[i + 2*pixels_per_image]) );  // Rayleigh image
      rayleigh_integral += energy_float;
      tmp[i + 2 * pixels_per_image] = energy_float;
    }
    for(i=0; i<pixels_per_image; i++)
    {
      energy_float = (float)( NORM * (double)(image[i + 3*pixels_per_image]) );  // Multiple-scatter image
      multiscatter_integral += energy_float;
      tmp[i + 3 * pixels_per_image] = energy_float;
    }       

    double energy_intergral = noScatter_integral + compton_integral + 
                              rayleigh_integral + multiscatter_integral;
    float noScatter_ratio = (float)(noScatter_integral / energy_intergral);
    float compton_ratio = (float)(compton_integral / energy_intergral);
    float rayleigh_ratio = (float)(rayleigh_integral / energy_intergral);
    float multiscatter_ratio = (float)(multiscatter_integral / energy_intergral);
    tmp[4 * pixels_per_image] = noScatter_ratio;
    tmp[4 * pixels_per_image + 1] = compton_ratio;
    tmp[4 * pixels_per_image + 2] = rayleigh_ratio;
    tmp[4 * pixels_per_image + 3] = multiscatter_ratio;
    
    file_binary_ptr.write(reinterpret_cast<char*>(tmp), (4 * pixels_per_image + 4) * sizeof(float));
    file_binary_ptr.close();
    delete[] tmp;
    
  }
  
  

    
  return 0;     // Report could return not 0 to continue the simulation...
}
///////////////////////////////////////////////////////////////////////////////




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
int report_voxels_dose(char* file_dose_output, int num_projections, struct voxel_struct* voxel_data, float2* voxel_mat_dens, ulonglong2* voxels_Edep, double time_total_MC_init_report, unsigned long long int total_histories, short int dose_ROI_x_min, short int dose_ROI_x_max, short int dose_ROI_y_min, short int dose_ROI_y_max, short int dose_ROI_z_min, short int dose_ROI_z_max, struct source_struct* source_data)
{

  mexPrintf("\n\n          *** VOXEL ROI DOSE TALLY REPORT ***\n\n");
    
  FILE* file_ptr = fopen(file_dose_output, "w");
  if (file_ptr==NULL)
  {
    mexPrintf("\n\n   !!fopen ERROR report_voxels_dose!! File %s can not be opened!!\n", file_dose_output);
    exit(-3);
  }    
    
  // -- Binary output:                                         // !!BINARY!!  
  char file_binary_mean[250], file_binary_sigma[250];
  strncpy (file_binary_mean, file_dose_output, 250);
  strcat(file_binary_mean,".raw");                     
  strncpy (file_binary_sigma, file_dose_output, 250);
  strcat(file_binary_sigma,"_2sigma.raw");    
  FILE* file_binary_mean_ptr  = fopen(file_binary_mean, "w");  // !!BINARY!!
  FILE* file_binary_sigma_ptr = fopen(file_binary_sigma, "w");       // !!BINARY!!
  if (file_binary_mean_ptr==NULL)
  {
    mexPrintf("\n\n   !!fopen ERROR report_voxels_dose!! Binary file %s can not be opened!!\n", file_dose_output);
    exit(-3);
  }
  
  int DX = dose_ROI_x_max - dose_ROI_x_min + 1,
      DY = dose_ROI_y_max - dose_ROI_y_min + 1,
      DZ = dose_ROI_z_max - dose_ROI_z_min + 1;           
      
  // -- Calculate the dose plane that will be output as ASCII text:
  int z_plane_dose = (int)(source_data[0].position.z * voxel_data->inv_voxel_size.z + 0.00001f);  // Select voxel plane at the level of the source, 1st projections
  if ( (z_plane_dose<dose_ROI_z_min) || (z_plane_dose>dose_ROI_z_max) )
    z_plane_dose = (dose_ROI_z_max+dose_ROI_z_min)/2;
  
  int z_plane_dose_ROI = z_plane_dose - dose_ROI_z_min;

  mexPrintf("              Reporting the 3D voxel dose distribution as binary floats in the .raw file, and the 2D dose for Z plane %d as ASCII text.\n", z_plane_dose);
//   mexPrintf("              Also reporting the dose to each material inside the input ROI adding the energy deposited in each individual voxel\n");
//   mexPrintf("              (these material dose results will be equal to the materials dose tally below if the ROI covers all the voxels).\n");
  
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "#     ***         MC-GPU, version 1.3 (http://code.google.com/p/mcgpu/)         ***\n");
  fprintf(file_ptr, "#     ***                                                                       ***\n");
  fprintf(file_ptr, "#     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "# \n");  
#ifdef USING_CUDA
  fprintf(file_ptr, "#  *** SIMULATION IN THE GPU USING CUDA ***\n");
#else
  fprintf(file_ptr, "#  *** SIMULATION IN THE CPU ***\n");
#endif
  fprintf(file_ptr, "#\n");
  
  
  // Report only one dose plane in ASCII, all the other data in binary only:

  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  3D dose deposition map (and dose uncertainty) created tallying the energy deposited by photons inside each voxel of the input geometry.\n");
  fprintf(file_ptr, "#  Electrons were not transported and therefore we are approximating that the dose is equal to the KERMA (energy released by the photons alone).\n");
  fprintf(file_ptr, "#  This approximation is acceptable when there is electronic equilibrium and when the range of the secondary electrons is shorter than the voxel size.\n");
  fprintf(file_ptr, "#  Usually the doses will be acceptable for photon energies below 1 MeV. The dose estimates may not be accurate at the interface of low density volumes.\n");
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  The 3D dose deposition is reported in binary form in the .raw files (data given as 32-bit floats). \n");
  fprintf(file_ptr, "#  To reduce the memory use and the reporting time this text output reports only the 2D dose at the Z plane at the level\n"); 
  fprintf(file_ptr, "#  of the source focal spot: z_coord = %d (z_coord in ROI = %d)\n", z_plane_dose, z_plane_dose_ROI);
  fprintf(file_ptr, "#\n");  
  fprintf(file_ptr, "#  The total dose deposited in each different material is reported to the standard output.\n");
  fprintf(file_ptr, "#  The dose is calculated adding the energy deposited in the individual voxels within the dose ROI and dividing by the total mass of the material in the ROI.\n");
  fprintf(file_ptr, "#\n");  
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  Voxel size:  %lf x %lf x %lf = %lf cm^3\n", 1.0/(double)(voxel_data->inv_voxel_size.x), 1.0/(double)(voxel_data->inv_voxel_size.y), 1.0/(double)(voxel_data->inv_voxel_size.z), 1.0/(double)(voxel_data->inv_voxel_size.x*voxel_data->inv_voxel_size.y*voxel_data->inv_voxel_size.z));
  fprintf(file_ptr, "#  Number of voxels in the reported region of interest (ROI) X, Y and Z:\n");
  fprintf(file_ptr, "#      %d  %d  %d\n", DX, DY, DZ);
  fprintf(file_ptr, "#  Coordinates of the ROI inside the voxel volume = X[%d,%d], Y[%d,%d], Z[%d,%d]\n", dose_ROI_x_min+1, dose_ROI_x_max+1, dose_ROI_y_min+1, dose_ROI_y_max+1, dose_ROI_z_min+1, dose_ROI_z_max+1);  // Show ROI with index=1 for the first voxel instead of 0.
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  Voxel dose units: eV/g per history\n");
  fprintf(file_ptr, "#  X rows given first, then Y, then Z. One blank line separates the different Y, and two blanks the Z values (GNUPLOT format).\n");
  fprintf(file_ptr, "#  The dose distribution is also reported with binary FLOAT values (.raw file) for easy visualization in ImageJ.\n");
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#    [DOSE]   [2*standard_deviation]\n");
  fprintf(file_ptr, "# =====================================\n");
  fflush(file_ptr);
  
  double voxel_dose, max_voxel_dose = -1.0, max_voxel_dose_std_dev = -1.0;
  
  int max_dose_voxel_geometry=0, max_voxel_dose_x=-1, max_voxel_dose_y=-1, max_voxel_dose_z=-1;
  unsigned long long int total_energy_deposited = 0;
  double inv_SCALE_eV = 1.0 / SCALE_eV,      // conversion to eV using the inverse of the constant used in the tally function (defined in the header file).         
                inv_N = 1.0 / (double)(total_histories*((unsigned long long int)num_projections));
                                
  register int i, j, k, voxel=0;
    
  double mat_Edep[MAX_MATERIALS], mat_Edep2[MAX_MATERIALS], mat_mass_ROI[MAX_MATERIALS];    // Arrays with the total energy, energy squared and mass of each material inside the ROI (mass and dose outside the ROI was not tallied).
  unsigned int mat_voxels[MAX_MATERIALS];
  for(i=0; i<MAX_MATERIALS; i++)
  {
     mat_Edep[i]  = 0.0;
     mat_Edep2[i] = 0.0;
     mat_mass_ROI[i]  = 0.0;
     mat_voxels[i]= 0;
  }
  
  double voxel_volume = 1.0 / ( ((double)voxel_data->inv_voxel_size.x) * ((double)voxel_data->inv_voxel_size.y) * ((double)voxel_data->inv_voxel_size.z) );
    
  for(k=0; k<DZ; k++)
  {
    for(j=0; j<DY; j++)
    {
      for(i=0; i<DX; i++)
      {
        register int voxel_geometry = (i+dose_ROI_x_min) + (j+dose_ROI_y_min)*voxel_data->num_voxels.x + (k+dose_ROI_z_min)*voxel_data->num_voxels.x*voxel_data->num_voxels.y;
        register double inv_voxel_mass = 1.0 / (voxel_mat_dens[voxel_geometry].y*voxel_volume);

        register int mat_number = (int)(voxel_mat_dens[voxel_geometry].x) - 1 ;  // Material number, starting at 0.
        mat_mass_ROI[mat_number]  += voxel_mat_dens[voxel_geometry].y*voxel_volume;   // Estimate mass and energy deposited in this material
        mat_Edep[mat_number]  += (double)voxels_Edep[voxel].x;        // Using doubles to avoid overflow
        mat_Edep2[mat_number] += (double)voxels_Edep[voxel].y;
        mat_voxels[mat_number]++;                                                // Count voxels made of this material
        
                
              // Optional code to eliminate dose deposited in air (first material).  Sometimes useful for visualization (dose to air irrelevant, noisy)
              //   if (voxel_mat_dens[voxel_geometry].x < 1.1f)
              //   {
              //     voxels_Edep[voxel].x = 0.0f;
              //     voxels_Edep[voxel].y = 0.0f;
              //   }
                
        // -- Convert total energy deposited to dose [eV/gram] per history:                        
        voxel_dose = ((double)voxels_Edep[voxel].x) * inv_N * inv_voxel_mass * inv_SCALE_eV;    // [dose == Edep * voxel_volume / voxel_density / N_hist]                      
        total_energy_deposited += voxels_Edep[voxel].x;

        register double voxel_std_dev = (((double)voxels_Edep[voxel].y) * inv_N * inv_SCALE_eV * inv_voxel_mass - voxel_dose*voxel_dose) * inv_N;   // [sigma = (<Edep^2> - <Edep>^2) / N_hist]
        if (voxel_std_dev>0.0)
          voxel_std_dev = sqrt(voxel_std_dev);
        
        if (voxel_dose > max_voxel_dose)
        {
          // Find the voxel that has the maximum dose:
          max_voxel_dose          = voxel_dose;
          max_voxel_dose_std_dev  = voxel_std_dev;
          max_voxel_dose_x        = i+dose_ROI_x_min;
          max_voxel_dose_y        = j+dose_ROI_y_min;
          max_voxel_dose_z        = k+dose_ROI_z_min;
          max_dose_voxel_geometry = voxel_geometry;          
        }
        
        // Report only one dose plane in ASCII:
        if (k == z_plane_dose_ROI) 
          fprintf(file_ptr, "%.6lf %.6lf\n", voxel_dose, 2.0*voxel_std_dev);        
        
        float voxel_dose_float  = (float)voxel_dose;         // After dividing by the number of histories I can report FLOAT bc the number of significant digits will be low.  
        float voxel_sigma_float = 2.0f * (float)(voxel_std_dev);
        
        fwrite(&voxel_dose_float,  sizeof(float), 1, file_binary_mean_ptr);    // Write dose data in a binary file that can be easyly open in imageJ.   !!BINARY!!
        fwrite(&voxel_sigma_float, sizeof(float), 1, file_binary_sigma_ptr);
        
        voxel++;
      }
      if (k == z_plane_dose_ROI) 
        fprintf(file_ptr, "\n");     // Separate Ys with an empty line for visualization with gnuplot.
    }
    if (k == z_plane_dose_ROI) 
      fprintf(file_ptr, "\n");     // Separate Zs.
  }

  
  fprintf(file_ptr, "#   ****** DOSE REPORT: TOTAL SIMULATION PERFORMANCE FOR ALL PROJECTIONS ******\n");
  fprintf(file_ptr, "#       Total number of simulated x rays: %lld\n", total_histories*((unsigned long long int)num_projections));
  fprintf(file_ptr, "#       Simulated x rays per projection:  %lld\n", total_histories);
  fprintf(file_ptr, "#       Total simulation time [s]:  %.2f\n", time_total_MC_init_report);
  if (time_total_MC_init_report>0.000001)
    fprintf(file_ptr, "#       Total speed [x-rays/s]:  %.2f\n", (double)(total_histories*((unsigned long long int)num_projections))/time_total_MC_init_report);

  
  fprintf(file_ptr, "\n#       Total energy absorved inside the dose ROI: %.5lf keV/hist\n\n", 0.001*((double)total_energy_deposited)*inv_N*inv_SCALE_eV);
  
  // Output data to standard input:
  mexPrintf("\n              Total energy absorved inside the dose deposition ROI: %.5lf keV/hist\n", 0.001*((double)total_energy_deposited)*inv_N*inv_SCALE_eV);
  register double voxel_mass_max_dose = voxel_volume*voxel_mat_dens[max_dose_voxel_geometry].y; 
  mexPrintf(  "              Maximum voxel dose (+-2 sigma): %lf +- %lf eV/g per history (E_dep_voxel=%lf eV/hist)\n", max_voxel_dose, max_voxel_dose_std_dev, (max_voxel_dose*voxel_mass_max_dose));
  mexPrintf(  "              for the voxel: material=%d, density=%.8f g/cm^3, voxel_mass=%.8lf g, voxel coord in geometry=(%d,%d,%d)\n\n", (int)voxel_mat_dens[max_dose_voxel_geometry].x, voxel_mat_dens[max_dose_voxel_geometry].y, voxel_mass_max_dose, max_voxel_dose_x, max_voxel_dose_y, max_voxel_dose_z);
  
  
  // -- Report dose deposited in each material:  
  mexPrintf("              Dose deposited in the different materials inside the input ROI computed post-processing the 3D voxel dose results:\n\n");
  mexPrintf("    [MATERIAL]  [DOSE_ROI, eV/g/hist]  [2*std_dev]  [Rel error 2*std_dev, %%]  [E_dep [eV/hist]  [MASS_ROI, g]  [NUM_VOXELS_ROI]\n");
  mexPrintf("   =============================================================================================================================\n");
  
  for(i=0; i<MAX_MATERIALS; i++)
  {
    if(mat_voxels[i]>0)   // Report only for materials found at least in 1 voxel of the input geometry (prevent dividing by 0 mass).
    {
      
      double Edep = mat_Edep[i] * inv_N * inv_SCALE_eV;    // [dose == Edep/Mass/N_hist]
      // !!DeBuG!! BUG in version 1.2: I have to divide by mass after computing the mean and sigma!!!
      // !!DeBuG!! WRONG code:  double material_dose = mat_Edep[i] * inv_N  * inv_SCALE_eV / mat_mass_ROI[i];    // [dose == Edep/Mass/N_hist]
      // !!DeBuG!! WRONG code:  double material_std_dev = (mat_Edep2[i] * inv_N  * inv_SCALE_eV / mat_mass_ROI[i] - material_dose*material_dose) * inv_N;   // [sigma^2 = (<Edep^2> - <Edep>^2) / N_hist]      
      
      double material_std_dev = (mat_Edep2[i] * inv_N - Edep*Edep) * inv_N;   // [sigma^2 = (<Edep^2> - <Edep>^2) / N_hist]   (mat_Edep2 not scaled by SCALE_eV in kernel to prevent overflow)
      if (material_std_dev>0.0)
        material_std_dev = sqrt(material_std_dev);
      
      double material_dose = Edep / mat_mass_ROI[i];
      material_std_dev = material_std_dev / mat_mass_ROI[i];
      
      double rel_diff = 0.0;
      if (material_dose>0.0)
        rel_diff = material_std_dev/material_dose;
    
      mexPrintf("\t%d\t%.5lf\t\t%.5lf\t\t%.2lf\t\t%.2lf\t\t%.5lf\t%u\n", (i+1), material_dose, 2.0*material_std_dev, (2.0*100.0*rel_diff), Edep, mat_mass_ROI[i], mat_voxels[i]);            
 
    }    
  }       
  mexPrintf("\n");
        
  
  fflush(stdout);          
  fclose(file_ptr);  // Close output file and flush stream
  fclose(file_binary_mean_ptr);
  fclose(file_binary_sigma_ptr);

  return 0;   // Report could return not 0 to continue the simulation...
}
///////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////
//! Report the tallied dose to each material number, accounting for different 
//! densities in different regions with the same material number. 
//!
//!       @param[in] num_projections   Number of projections simulated
//!       @param[in] total_histories   Total number of x-rays simulated per projection
//!       @param[out] density_nominal   Array with the nominal densities of materials given in the input file; -1 for materials not defined. Used to report only defined materials.
//!       @param[in] materials_dose   Tallied dose and dose^2 arrays
////////////////////////////////////////////////////////////////////////////////
int report_materials_dose(int num_projections, unsigned long long int total_histories, float *density_nominal, ulonglong2 *materials_dose, double *mass_materials)  // !!tally_materials_dose!!
{

  mexPrintf("\n\n          *** MATERIALS TOTAL DOSE TALLY REPORT ***\n\n");  
  mexPrintf("              Dose deposited in each material defined in the input file (tallied directly per material, not per voxel):\n");
  mexPrintf("              The results of this tally should be equal to the voxel tally doses for an ROI covering all voxels.\n\n");
  mexPrintf("    [MAT]  [DOSE, eV/g/hist]  [2*std_dev]  [Rel_error 2*std_dev, %%]  [E_dep [eV/hist]  [MASS_TOTAL, g]\n");
  mexPrintf("   ====================================================================================================\n");
  
  double dose, Edep, std_dev, rel_diff, inv_N = 1.0 / (double)(total_histories*((unsigned long long int)num_projections));
  int i, flag=0, max_mat=0;
  for(i=0; i<MAX_MATERIALS; i++)
  {
    if (density_nominal[i]<0.0f)
      break;  // Skip report for materials not defined in the input file
    
    Edep    = ((double)materials_dose[i].x) / SCALE_eV * inv_N;
    
    std_dev = sqrt( (((double)materials_dose[i].y)*inv_N - Edep*Edep) * inv_N );   // [sigma^2 = (<Edep^2> - <Edep>^2) / N_hist]   (not scaling "materials_dose[i].y" by SCALE_eV in kernel to prevent overflow).
    
    if (Edep>0.0)
      rel_diff = std_dev/Edep;
    else
      rel_diff = 0.0;

    dose    = Edep / mass_materials[i];
    std_dev = std_dev / mass_materials[i];
    
    mexPrintf("\t%d\t%.5lf\t\t%.5lf\t\t%.2lf\t\t%.2lf\t\t%.5lf\n", (i+1), dose, 2.0*std_dev, 2.0*100.0*rel_diff, Edep, mass_materials[i]);
    
       // mexPrintf("\t%d\t%.5lf\t\t%.5lf\t\t%.2lf\t\t%.5lf\t\t%.5lf\t\t\t%llu,\t\t%llu\n", (i+1), dose, 2.0*std_dev, 2.0*100.0*rel_diff, Edep, mass_materials[i],  materials_dose[i].x, materials_dose[i].y);  //!!DeBuG!! VERBOSE output: counters not divided by num histories
    
    
    if (materials_dose[i].x>1e16 || dose!=fabs(dose) || std_dev!=fabs(std_dev))  // !!DeBuG!!  Try to detect a possible overflow in any material: large counter or negative, nan value
    {
      flag = 1;
      if (materials_dose[i].x>materials_dose[max_mat].x) 
        max_mat = i;
    }
  }
  
  if (flag!=0)    // !!DeBuG!! Try to detect a possible overflow: large counter or negative, nan value. The value of SCALE_eV can be reduced to prevent this overflow in some cases.
  {
    mexPrintf("\n     WARNING: it is possible that the unsigned long long int counter used to tally the standard deviation overflowed (>2^64).\n");  // !!DeBuG!! 
    mexPrintf("              The standard deviation may be incorrectly measured, but it will surely be very small (<< 1%%).\n");
    mexPrintf("              Max counter (mat=%d): E_dep = %llu , E_dep^2 = %llu\n\n", max_mat+1, materials_dose[max_mat].x, materials_dose[max_mat].y);
  }
  fflush(stdout);  
  return 0;
}


///////////////////////////////////////////////////////////////////////////////