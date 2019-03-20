/* Test and timing harness program for developing a multichannel
   multikernel convolution (as used in deep learning networks)

   Note there are some simplifications around this implementation,
   in particular with respect to computing the convolution at edge
   pixels of the image.

   Author: David Gregg
   Date:   February 2019

   Version 1.5 : Modified the code so that the input and kernel
                 are tensors of 16-bit integer values

   Version 1.4 : Modified the random generator to reduce the range
                 of generated values;

   Version 1.3 : Fixed which loop variables were being incremented
                 in write_out();
                 Fixed dimensions of output and control_output
                 matrices in main function

   Version 1.2 : Changed distribution of test data to (hopefully)
                 eliminate random walk of floating point error;
                 Also introduced checks to restrict kernel-order to
                 a small set of values

   Version 1.1 : Fixed bug in code to create 4d matrix
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <x86intrin.h>
#include <sys/sysinfo.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)


/* write 3d matrix to stdout */
void write_out(int16_t *** a, int dim0, int dim1, int dim2)
{
  int i, j, k;

  for ( i = 0; i < dim0; i++ ) {
    printf("Outer dimension number %d\n", i);
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2 - 1; k++ ) {
        printf("%d, ", a[i][j][k]);
      }
      // print end of line
      printf("%f\n", a[i][j][dim2-1]);
    }
  }
}


/* create new empty 4d float matrix */
float **** new_empty_4d_matrix_float(int dim0, int dim1, int dim2, int dim3)
{
  float **** result = malloc(dim0 * sizeof(float***));
  float *** mat1 = malloc(dim0 * dim1 * sizeof(float**));
  float ** mat2 = malloc(dim0 * dim1 * dim2 * sizeof(float*));
  float * mat3 = malloc(dim0 * dim1 * dim2 *dim3 * sizeof(float));
  int i, j, k;


  for ( i = 0; i < dim0; i++ ) {
    result[i] = &(mat1[i*dim1]);
    for ( j = 0; j < dim1; j++ ) {
      result[i][j] = &(mat2[i*dim1*dim2 + j*dim2]);
      for ( k = 0; k < dim2; k++ ) {
        result[i][j][k] = &(mat3[i*dim1*dim2*dim3+j*dim2*dim3+k*dim3]);
      }
    }
  }

  return result;
}

/* create new empty 3d matrix */
float *** new_empty_3d_matrix_float(int dim0, int dim1, int dim2)
{
  float **** mat4d;
  float *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix_float(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* create new empty 4d int16_t matrix */
int16_t **** new_empty_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
  int16_t **** result = malloc(dim0 * sizeof(int16_t***));
  int16_t *** mat1 = malloc(dim0 * dim1 * sizeof(int16_t**));
  int16_t ** mat2 = malloc(dim0 * dim1 * dim2 * sizeof(int16_t*));
  int16_t * mat3 = malloc(dim0 * dim1 * dim2 *dim3 * sizeof(int16_t));
  int i, j, k;


  for ( i = 0; i < dim0; i++ ) {
    result[i] = &(mat1[i*dim1]);
    for ( j = 0; j < dim1; j++ ) {
      result[i][j] = &(mat2[i*dim1*dim2 + j*dim2]);
      for ( k = 0; k < dim2; k++ ) {
        result[i][j][k] = &(mat3[i*dim1*dim2*dim3+j*dim2*dim3+k*dim3]);
      }
    }
  }

  return result;
}

/* create new empty 3d matrix */
int16_t *** new_empty_3d_matrix_int16(int dim0, int dim1, int dim2)
{
  int16_t **** mat4d;
  int16_t *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix_int16(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* take a copy of the matrix and return in a newly allocated matrix */
int16_t **** copy_4d_matrix(int16_t **** source_matrix, int dim0,
                            int dim1, int dim2, int dim3)
{
  int i, j, k, l;
  int16_t **** result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        for ( l = 0; l < dim3; l++ ) {
          result[i][j][k][l] = source_matrix[i][j][k][l];
        }
      }
    }
  }
  return result;
}

/* create a matrix and fill it with random numbers */
int16_t **** gen_random_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
int16_t **** result;
int i, j, k, l;
struct timeval seedtime;
  int seed;

  result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  const int range = 1 << 10; // 2^10
  //const int bias = 1 << 16; // 2^16
  int16_t offset = 0.0;
  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        for ( l = 0; l < dim3; l++ ) {
          // generate uniform random integer with mean of zero
          long long rand = random();
          // now cut down the range and bias the mean to reduce
          // the likelihood of large floating point round-off errors
          int reduced_range = (rand % range);
          result[i][j][k][l] = reduced_range;
        }
      }
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
int16_t *** gen_random_3d_matrix_int16(int dim0, int dim1, int dim2)
{
  int16_t **** mat4d;
  int16_t *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = gen_random_4d_matrix_int16(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(float *** result, float *** control,
                  int dim0, int dim1, int dim2)
{
  int i, j, k;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  //printf("SAD\n");

  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        double diff = fabs(control[i][j][k] - result[i][j][k]);
        assert( diff >= 0.0 );
        sum_abs_diff = sum_abs_diff + diff;
      }
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
            sum_abs_diff, EPSILON);
  }
  else {
    printf("COMMENT: sum of absolute differences (%f)  within acceptable range (%f)\n", sum_abs_diff, EPSILON);
  }
}

/* the slow but correct version of matmul written by David */
void multichannel_conv(int16_t *** image, int16_t **** kernels,
                        float *** output, int width, int height,
                        int nchannels, int nkernels, int kernel_order)
{
  int h, w, x, y, c, m;

  for ( m = 0; m < nkernels; m++ ) {
    for ( w = 0; w < width; w++ ) {
      for ( h = 0; h < height; h++ ) {
        double sum = 0.0;
        for ( c = 0; c < nchannels; c++ ) {
          for ( x = 0; x < kernel_order; x++) {
            for ( y = 0; y < kernel_order; y++ ) {
              sum += (double) image[w+x][h+y][c] * (double) kernels[m][c][x][y];
            }
          }
        }
        output[m][w][h] = (float) sum;
      }
    }
  }
}

/* the fast version of matmul written by the team */

/* pthread function for independent loops inside */

struct kernel_data {
  int16_t ***image;
  int16_t ****kernels;
  float ***output;
  int width;
  int height;
  int nchannels;
  int kernel_order;
  int start_kernel;
  int nkernels;
};

void *nkernel_calcs(void *calc_data)
{
  struct kernel_data *data = calc_data;
  int16_t ****kernels = data->kernels;
  int16_t ***image = data->image;
  float ***output = data->output;
  int width = data->width;
  int height = data->height;
  int nchannels = data->nchannels;
  int kernel_order = data->kernel_order;
  int start_kernel = data->start_kernel;
  int nkernels = data->nkernels;
  int max_kernel = start_kernel + nkernels;
  printf("Starting thread to calc kernal %d to %d\n", start_kernel, start_kernel+nkernels-1);

  int h, w, x, y, c, m;
  __m128 kernel_val[kernel_order];
  __m128 image_val[kernel_order];

  if (kernel_order == 1) {
    for ( m = start_kernel; m < max_kernel; m++ ) {
      for ( w = 0; w < width; w++ ) {
        for ( h = 0; h < height; h++ ) {
          double sum = 0.0;
          for ( c = 0; c < nchannels; c++ ) {
            sum += (float) image[w][h][c] * (float) kernels[m][c][0][0];
          }
          output[m][w][h] = (float) sum;
        }
      }
    }
  } else if (kernel_order == 3) {

    for ( m = start_kernel; m < max_kernel; m++ ) {
      double *temp_doubles = malloc(width * height * sizeof(double));
      for ( c = 0; c < nchannels; c++ ) {
        //load the entire kernel into the vectors
        kernel_val[0] = _mm_set_ps((float) kernels[m][c][0][0], (float) kernels[m][c][0][1], (float) kernels[m][c][0][2],0.0);
        kernel_val[1] = _mm_set_ps((float) kernels[m][c][1][0], (float) kernels[m][c][1][1], (float) kernels[m][c][1][2],0.0);
        kernel_val[2] = _mm_set_ps((float) kernels[m][c][2][0], (float) kernels[m][c][2][1], (float) kernels[m][c][2][2],0.0);
        for ( w = 0; w < width; w++ ) {
          for ( h = 0; h < height; h++ ) {
            double sum = 0.0;
            if (c != 0) {sum = temp_doubles[w*height + h];}
            if (h == 0) {
              //set up the intial vectors for the first element
              image_val[0] = _mm_set_ps((float) image[w][h][c], (float) image[w][h+1][c], (float) image[w][h+2][c],0.0);
              image_val[1] = _mm_set_ps((float) image[w+1][h][c], (float) image[w+1][h+1][c], (float) image[w+1][h+2][c],0.0);
              image_val[2] = _mm_set_ps((float) image[w+2][h][c], (float) image[w+2][h+1][c], (float) image[w+2][h+2][c],0.0);
            } else {
              //crawl over the values of the vectors as the image is gone through
              image_val[0] = _mm_shuffle_ps(_mm_set_ss((float) image[w][h+2][c]), image_val[0], 0x91); //10010001
              image_val[1] = _mm_shuffle_ps(_mm_set_ss((float) image[w+1][h+2][c]), image_val[1], 0x91);
              image_val[2] = _mm_shuffle_ps(_mm_set_ss((float) image[w+2][h+2][c]), image_val[2], 0x91);
            }
            for (int row = 0; row < 3; row++){
              __m128 values = _mm_mul_ps(image_val[row],kernel_val[row]);
              values = _mm_hadd_ps(values, values);
              values = _mm_hadd_ps(values, values);
              sum += (double) _mm_cvtss_f32(values);
            }
            temp_doubles[w*height + h] = sum;
          }
        }
      }
      for ( w = 0; w < width; w++ ) {
        for ( h = 0; h < height; h++ ) {
          output[m][w][h] = (float) temp_doubles[w*height + h];
        }
      }
    }
  } else {
    for ( m = start_kernel; m < max_kernel; m++ ) {
      double *temp_doubles = malloc(width * height * sizeof(double));
      for ( c = 0; c < nchannels; c++ ) {
        //load as much of the kernel into SSE vectors as possible
        for (int kern = 0; kern < kernel_order; kern++){
          kernel_val[kern] = _mm_set_ps((float) kernels[m][c][kern][0], (float) kernels[m][c][kern][1], (float) kernels[m][c][kern][2], (float) kernels[m][c][kern][3]); 
        }
        for ( w = 0; w < width; w++ ) {
          for ( h = 0; h < height; h++ ) {
            double sum = 0.0;
            if (c != 0) {sum = temp_doubles[w*height + h];}
            if (h == 0) {
              //set up the intial vectors for the first element
              for (int kern = 0; kern < kernel_order; kern++){
                image_val[kern] = _mm_set_ps((float) image[w+kern][h][c], (float) image[w+kern][h+1][c], (float) image[w+kern][h+2][c],(float) image[w+kern][h+3][c]);
              }
            } else {
              //crawl over the values of the vectors as the image is gone through
              for (int kern = 0; kern < kernel_order; kern++){
                __m128 temp = _mm_set_ss((float) image[w+kern][h+3][c]);
                image_val[kern] = _mm_move_ss(_mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(image_val[kern]), 4)), temp);
              }
            }
            for (int row = 0; row < kernel_order; row++){
              //calculate the values in the SSE vectors
              __m128 values = _mm_mul_ps(image_val[row],kernel_val[row]);
              values = _mm_hadd_ps(values, values);
              values = _mm_hadd_ps(values, values);
              sum += (double) _mm_cvtss_f32(values);
              //add up the values not covered by SSE
              for (int remain = 4; remain < kernel_order; remain++) {
                sum += (double) image[w+row][h+remain][c] * (double) kernels[m][c][row][remain];
              }
            }
            temp_doubles[w*height + h] = sum;
          }
        }
      }
      for ( w = 0; w < width; w++ ) {
        for ( h = 0; h < height; h++ ) {
          output[m][w][h] = (float) temp_doubles[w*height + h];
        }
      }
    }
  }
}

void team_conv(int16_t *** image, int16_t **** kernels, float *** output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  // pthread_sse2

  int m;
  int nprocs = get_nprocs();
  int nthreads = nkernels > nprocs ? nprocs : nkernels;
  printf("Starting %d threads...\n", nthreads);
  int kernelsperthread = nkernels / nthreads;
  int extrakernels = nkernels % nthreads;
  int kernels_done = 0;

  pthread_t kernel_calcs[nthreads];
  for ( m = 0; m < nthreads; m++ ) {
    int threadkernels = extrakernels-- > 0 ? kernelsperthread + 1 : kernelsperthread;
    
    struct kernel_data *data = malloc(sizeof(struct kernel_data));
    data->image = image;
    data->kernels = kernels;
    data->output = output;
    data->width = width;
    data->height = height;
    data->nchannels = nchannels;
    data->kernel_order = kernel_order;
    data->start_kernel = kernels_done;
    data->nkernels = threadkernels;

    pthread_create(&kernel_calcs[m], NULL, nkernel_calcs, (void *)data);
    kernels_done += threadkernels;
    
  }

  for (m = 0; m < nthreads; m++) {
    pthread_join(kernel_calcs[m], NULL);
  }
  
  //default:
  //multichannel_conv(image, kernels, output, width,
  //                  height, nchannels, nkernels, kernel_order);
}

int main(int argc, char ** argv)
{
  //float image[W][H][C];
  //float kernels[M][C][K][K];
  //float output[M][W][H];

  int16_t *** image, **** kernels;
  float *** control_output, *** output;
  long long mul_time;
  int width, height, kernel_order, nchannels, nkernels;
  struct timeval start_time;
  struct timeval stop_time;

  if ( argc != 6 ) {
    fprintf(stderr, "Usage: conv-harness <image_width> <image_height> <kernel_order> <number of channels> <number of kernels>\n");
    exit(1);
  }
  else {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    kernel_order = atoi(argv[3]);
    nchannels = atoi(argv[4]);
    nkernels = atoi(argv[5]);
  }
  switch ( kernel_order ) {
  case 1:
  case 3:
  case 5:
  case 7: break;
  default:
    fprintf(stderr, "FATAL: kernel_order must be 1, 3, 5 or 7, not %d\n",
            kernel_order);
    exit(1);
  }

  /* allocate the matrices */
  image = gen_random_3d_matrix_int16(width+kernel_order, height + kernel_order,
                               nchannels);
  kernels = gen_random_4d_matrix_int16(nkernels, nchannels, kernel_order, kernel_order);
  output = new_empty_3d_matrix_float(nkernels, width, height);
  control_output = new_empty_3d_matrix_float(nkernels, width, height);

  //DEBUGGING(write_out(A, a_dim1, a_dim2));

  /* use a simple multichannel convolution routine to produce control result */
  multichannel_conv(image, kernels, control_output, width,
                    height, nchannels, nkernels, kernel_order);

  /* record starting time of team's code*/
  gettimeofday(&start_time, NULL);

  /* perform student team's multichannel convolution */
  team_conv(image, kernels, output, width,
                    height, nchannels, nkernels, kernel_order);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Team conv time: %lld microseconds\n", mul_time);

  DEBUGGING(write_out(output, nkernels, width, height));

  /* now check that the team's multichannel convolution routine
     gives the same answer as the known working version */
  check_result(output, control_output, nkernels, width, height);

  return 0;
}
