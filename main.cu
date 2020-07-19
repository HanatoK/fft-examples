#include <cmath>
#include <cstdio>
#include <fftw3.h>
#include <cstring>

#include "cufft.h"
#include "dft.h"

template <typename T>
class InputData {
public:
  InputData(int nx, int ny, double factor = 2.0) {
    m_nx = nx;
    m_ny = ny;
    m_data = new T[m_nx*m_ny];
    prepareData(factor);
  }
  void prepareData(double factor) {
    // x ranges from -1 to 1
    // y ranges from -1 to 1
    const T width_x = 2.0/m_nx;
    const T width_y = 2.0/m_ny;
    const T tmp_x = -1.0+0.5*width_x;
    const T tmp_y = -1.0+0.5*width_y;
    for (int i = 0; i < m_nx; ++i) {
      for (int j = 0; j < m_ny; ++j) {
//         m_data[j*m_nx+i] = std::sin(tmp_x+width_x*i) + std::cos(tmp_y+width_y*j);
        m_data[j*m_nx+i] = (tmp_x+width_x*i)*(tmp_y+factor*width_y*j);
      }
    }
  }
  ~InputData() {
    delete[] m_data;
  }
  int size() const {
    return m_nx*m_ny;
  }
  void writeToFile(const char* filename) const {
    FILE* handle = fopen(filename, "w");
    for (int j = 0; j < m_ny; ++j) {
      for (int i = 0; i < m_nx; ++i) {
        fprintf(handle, "%15.5f", m_data[j*m_nx+i]);
      }
      fprintf(handle, "\n");
    }
    fclose(handle);
  }
  vector<vector<double>> getMatrix2D() const {
      vector<vector<double>> result(m_ny, vector<double>(m_nx, 0));
      for (int j = 0; j < m_ny; ++j) {
          T* data_begin = m_data + j * m_nx;
          T* data_end = data_begin + m_nx;
          std::copy(data_begin, data_end, result[j].begin());
      }
      return result;
  }
  int m_nx;
  int m_ny;
  T* m_data;
};

void writeToFile(float2* array, const char* filename, int X, int Y) {
  char* fn_real;
  char* fn_imag;
  const char* real_suffix = ".real";
  const char* imag_suffix = ".imag";
  fn_real = new char[strlen(filename)+strlen(real_suffix)+1];
  fn_imag = new char[strlen(filename)+strlen(imag_suffix)+1];
  sprintf(fn_real, "%s%s", filename, real_suffix);
  sprintf(fn_imag, "%s%s", filename, imag_suffix);
  FILE* handle_real = fopen(fn_real, "w");
  FILE* handle_imag = fopen(fn_imag, "w");
  for (int j = 0; j < Y; ++j) {
    for (int i = 0; i < X; ++i) {
      fprintf(handle_real, "%15.5f", array[j*X+i].x);
      fprintf(handle_imag, "%15.5f", array[j*X+i].y);
    }
    fprintf(handle_real, "\n");
    fprintf(handle_imag, "\n");
  }
  fclose(handle_real);
  fclose(handle_imag);
  delete[] fn_real;
  delete[] fn_imag;
}

void writeToFile(double2* array, const char* filename, int X, int Y) {
  char* fn_real;
  char* fn_imag;
  const char* real_suffix = ".real";
  const char* imag_suffix = ".imag";
  fn_real = new char[strlen(filename)+strlen(real_suffix)+1];
  fn_imag = new char[strlen(filename)+strlen(imag_suffix)+1];
  sprintf(fn_real, "%s%s", filename, real_suffix);
  sprintf(fn_imag, "%s%s", filename, imag_suffix);
  FILE* handle_real = fopen(fn_real, "w");
  FILE* handle_imag = fopen(fn_imag, "w");
  for (int j = 0; j < Y; ++j) {
    for (int i = 0; i < X; ++i) {
      fprintf(handle_real, "%15.5f", array[j*X+i].x);
      fprintf(handle_imag, "%15.5f", array[j*X+i].y);
    }
    fprintf(handle_real, "\n");
    fprintf(handle_imag, "\n");
  }
  fclose(handle_real);
  fclose(handle_imag);
  delete[] fn_real;
  delete[] fn_imag;
}

void writeToFile(fftw_complex* array, const char* filename, int X, int Y) {
  char* fn_real;
  char* fn_imag;
  const char* real_suffix = ".real";
  const char* imag_suffix = ".imag";
  fn_real = new char[strlen(filename)+strlen(real_suffix)+1];
  fn_imag = new char[strlen(filename)+strlen(imag_suffix)+1];
  sprintf(fn_real, "%s%s", filename, real_suffix);
  sprintf(fn_imag, "%s%s", filename, imag_suffix);
  FILE* handle_real = fopen(fn_real, "w");
  FILE* handle_imag = fopen(fn_imag, "w");
  for (int j = 0; j < Y; ++j) {
    for (int i = 0; i < X; ++i) {
      fprintf(handle_real, "%15.5f", array[j*X+i][0]);
      fprintf(handle_imag, "%15.5f", array[j*X+i][1]);
    }
    fprintf(handle_real, "\n");
    fprintf(handle_imag, "\n");
  }
  fclose(handle_real);
  fclose(handle_imag);
  delete[] fn_real;
  delete[] fn_imag;
}

template <typename T>
void writeToFile(T* array, const char* filename, int X, int Y) {
  FILE* handle = fopen(filename, "w");
  for (int j = 0; j < Y; ++j) {
    for (int i = 0; i < X; ++i) {
      fprintf(handle, "%15.5f", array[j*X+i]);
    }
    fprintf(handle, "\n");
  }
  fclose(handle);
}

void cpu_fft(int filename_index, double factor = 2.0) {
  const std::string inputdata_filename = std::string{"fftw_input"} + std::to_string(filename_index) + std::string{".dat"};
  const std::string r2c_output_filename = std::string{"fftw_output"} + std::to_string(filename_index) + std::string{"_r2c.dat"};
  const std::string c2r_output_filename = std::string{"fftw_output"} + std::to_string(filename_index) + std::string{"_c2r.dat"};
  InputData<double> input(128, 64, factor);
  input.writeToFile(inputdata_filename.c_str());
  fftw_complex *out;
  out = new fftw_complex[(input.m_nx/2+1)*input.m_ny];
  fftw_plan plan2d_forward = fftw_plan_dft_r2c_2d(input.m_ny, input.m_nx, input.m_data, out, FFTW_ESTIMATE);
  fftw_execute(plan2d_forward);
  writeToFile(out, r2c_output_filename.c_str(), input.m_nx/2+1, input.m_ny);
  fftw_plan plan2d_backward = fftw_plan_dft_c2r_2d(input.m_ny, input.m_nx, out, input.m_data, FFTW_ESTIMATE);
  fftw_execute(plan2d_backward);
  fftw_destroy_plan(plan2d_forward);
  fftw_destroy_plan(plan2d_backward);
  input.writeToFile(c2r_output_filename.c_str());
  delete[] out;
}

void gpu_fft() {
  const int nbatch = 2;
  InputData<double> input1(128, 64, 2.0);
  InputData<double> input2(128, 64, 1.5);
  const int stride = 128;
  double *d_input, *h_input;
  h_input = new double[input1.size()+stride+input2.size()];
  cudaMalloc(&d_input, (input1.size()+stride+input2.size())*sizeof(double));
  cudaMemcpy(d_input, input1.m_data, input1.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input+input1.size()+stride, input2.m_data, input2.size()*sizeof(double), cudaMemcpyHostToDevice);
  cufftHandle planManyForward, planManyBackward;
  cufftCreate(&planManyForward);
  cufftCreate(&planManyBackward);
  size_t r2cWorkSize, c2rWorkSize;
  int nIn[] = {input1.m_ny, input1.m_nx};
  int nOut[] = {input1.m_ny, input1.m_nx/2+1};
  const int idist = input1.m_ny*input1.m_nx+stride;
  const int odist = (input1.m_nx/2+1)*input1.m_ny;
  cufftMakePlanMany(planManyForward, // cufftHandle returned by cufftCreate
                    2,               // dimensionality
                    nIn,             // size of each dimension
                    nIn,             // inembed should be larger than or equal to nIn
                    1,               // istride
                    idist,           // idist
                    nOut,            // onembed
                    1,               // ostride
                    odist,           // odist
                    CUFFT_D2Z,       // fft type
                    nbatch,          // number of transformations
                    &r2cWorkSize);
  cufftMakePlanMany(planManyBackward,
                    2,
                    nIn,
                    nOut,
                    1,
                    odist,
                    nIn,
                    1,
                    idist,
                    CUFFT_Z2D,
                    nbatch,
                    &c2rWorkSize);
  cufftDoubleComplex *d_transform;
  cudaMalloc(&d_transform, input1.m_ny*(input1.m_nx/2+1)*nbatch*sizeof(cufftDoubleComplex));
  cufftExecD2Z(planManyForward, d_input, d_transform);
  double2* h_output;
  h_output = new double2[input1.m_ny*(input1.m_nx/2+1)*nbatch];
  cudaMemcpy(h_output, (double2*)d_transform, input1.m_ny*(input1.m_nx/2+1)*nbatch*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
  cufftExecZ2D(planManyBackward, d_transform, d_input);
  cudaMemcpy(h_input, d_input, (input1.size()+stride+input2.size())*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_transform);
  cufftDestroy(planManyForward);
  cufftDestroy(planManyBackward);
  writeToFile(h_output, "cufft_output1_r2c.dat", input1.m_nx/2+1, input1.m_ny);
  writeToFile(h_output+odist, "cufft_output2_r2c.dat", input1.m_nx/2+1, input1.m_ny);
  writeToFile(h_input, "cufft_output1_c2r.dat", input1.m_nx, input1.m_ny);
  writeToFile(h_input+idist, "cufft_output2_c2r.dat", input1.m_nx, input1.m_ny);
  delete[] h_input;
  delete[] h_output;
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

void gpu_fft_2() {
  //TODO
}

void me_fft(int filename_index, double factor = 2.0) {
  const std::string inputdata_filename = std::string{"me_fft_input"} + std::to_string(filename_index) + std::string{".dat"};
  const std::string r2c_output_filename = std::string{"me_fft_output"} + std::to_string(filename_index) + std::string{"_r2c.dat"};
  const std::string c2r_output_filename = std::string{"me_fft_output"} + std::to_string(filename_index) + std::string{"_c2r.dat"};
  InputData<double> input(128, 64, factor);
  const auto inputMatrix = input.getMatrix2D();
  writeToFile(inputMatrix, inputdata_filename.c_str());
  vector<vector<complex<double>>> fft_result = fastFourierTransform2D(inputMatrix);
  writeToFile(fft_result, r2c_output_filename.c_str());
  vector<vector<complex<double>>> rfft_result = reverseFastFourierTransform2D(fft_result);
  writeToFile(rfft_result, c2r_output_filename.c_str());
}

void me_dft(int filename_index, double factor = 2.0) {
  const std::string inputdata_filename = std::string{"me_dft_input"} + std::to_string(filename_index) + std::string{".dat"};
  const std::string r2c_output_filename = std::string{"me_dft_output"} + std::to_string(filename_index) + std::string{"_r2c.dat"};
  const std::string c2r_output_filename = std::string{"me_dft_output"} + std::to_string(filename_index) + std::string{"_c2r.dat"};
  InputData<double> input(128, 64, factor);
  const auto inputMatrix = input.getMatrix2D();
  writeToFile(inputMatrix, inputdata_filename.c_str());
  vector<vector<complex<double>>> fft_result = fourierTransform2D(inputMatrix);
  writeToFile(fft_result, r2c_output_filename.c_str());
  vector<vector<complex<double>>> rfft_result = reverseFourierTransform2D(fft_result);
  writeToFile(rfft_result, c2r_output_filename.c_str());
}

void cpu_fft_2() {
  // prepare two inputs
  InputData<double> input1(128, 64, 2.0);
  InputData<double> input2(128, 64, 1.5);
  input1.writeToFile("fftw_input1.dat");
  input2.writeToFile("fftw_input2.dat");
  // prepare array to combine the inputs
  const size_t stride = 100;
  double* inputAll = new double[input1.size()+input2.size()+stride];
  const double* i1begin = input1.m_data;
  const double* i1end = input1.m_data+input1.size();
  const double* i2begin = input2.m_data;
  const double* i2end = input2.m_data+input2.size();
  // copy data
  std::copy(i1begin, i1end, inputAll);
  std::copy(i2begin, i2end, inputAll+input1.size()+stride);
  // setup output array
  fftw_complex *out;
  out = new fftw_complex[(input1.m_nx/2+1)*input1.m_ny*2];
  // input size array
  const int nIn[] = {input1.m_ny, input1.m_nx};
  const int idist = input1.m_ny*input1.m_nx+stride;
  const int odist = (input1.m_nx/2+1)*input1.m_ny;
  fftw_plan plan2d_many_forward = fftw_plan_many_dft_r2c(
    2,        // rank or dimensionality of FFT
    nIn,      // size in each dimension of FFT
    2,        // number of transformations
    inputAll, // input array
    NULL,     // inembed should be larger than or equal to nIn
    1,        // input stride = 1
    idist,    // idist (padding)
    out,      // output array
    NULL,     // onembed should be larger than or equal to nOut
    1,        // output stride = 1
    odist,    // no padding in the output array
    FFTW_ESTIMATE);
  fftw_execute(plan2d_many_forward);
  writeToFile(out, "fftw_many_output1_r2c.dat", input1.m_nx/2+1, input1.m_ny);
  writeToFile(out+odist, "fftw_many_output2_r2c.dat", input2.m_nx/2+1, input2.m_ny);
  fftw_plan plan2d_many_backward = fftw_plan_many_dft_c2r(
    2,
    nIn,
    2,
    out,
    NULL,
    1,
    odist,
    inputAll,
    NULL,
    1,
    idist,
    FFTW_ESTIMATE);
  fftw_execute(plan2d_many_backward);
  writeToFile(inputAll, "fftw_many_output1_c2r.dat", input1.m_nx, input1.m_ny);
  writeToFile(inputAll+idist, "fftw_many_output2_c2r.dat", input1.m_nx, input1.m_ny);
  delete[] out;
  delete[] inputAll;
  fftw_destroy_plan(plan2d_many_forward);
  fftw_destroy_plan(plan2d_many_backward);
}

int main() {
  cpu_fft(1, 2.0);
  cpu_fft(2, 1.5);
  gpu_fft();
  me_dft(1, 2.0);
  me_dft(2, 1.5);
  me_fft(1, 2.0);
  me_fft(2, 1.5);
  cpu_fft_2();
  return 0;
}
