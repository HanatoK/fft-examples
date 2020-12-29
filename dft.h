#define _USE_MATH_DEFINES
#include <complex>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <cstdio>

using std::complex;
using std::vector;
using std::cout;

const size_t FFT_MIN_N = size_t(std::pow(2,4));
const size_t FFT_RADIX_3_MIN_N = size_t(std::pow(3,3));

vector<complex<double>> fourierTransform(const vector<complex<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input.size();
    vector<complex<double>> result(N);
    for (size_t k = 0; k < N; ++k) {
        complex<double> rN(0, 0);
        for (size_t j = 0; j < N; ++j) {
            const complex<double> tmp(0, -1.0 * j * 2.0 * M_PI * k / N);
            rN += input[j] * std::exp(tmp);
        }
        result[k] = rN;
    }
    return result;
}

complex<double> fastFourierTransform(const vector<complex<double>>& input,
                                     const vector<complex<double>>& factors,
                                     const complex<double>& factor2,
                                     size_t k, size_t read_start, size_t read_stride, size_t N,
                                     complex<double>& even_saved, complex<double>& odd_saved) {
    using namespace std::complex_literals;
    complex<double> rN(0, 0);
    complex<double> rN_odd(0, 0);
    complex<double> rN_even(0, 0);
    if (N % 2 != 0) {
        for (size_t j = 0; j < N; ++j) {
            const complex<double> tmp(0, -1.0 * j * 2.0 * M_PI * k / N);
            rN += input[read_start + j * read_stride]
                * std::exp(tmp);
        }
        return rN;
    } else {
        if (N == FFT_MIN_N) {
            for (size_t j = 0; j < N / 2; ++j) {
                rN_even += input[read_start + 2 * j * read_stride] * factors[j];
                rN_odd  += input[read_start + (2 * j + 1) * read_stride] * factors[j];
            }
        } else {
            const complex<double> new_factor2 = factor2 * factor2;
            rN_even = fastFourierTransform(input, factors, new_factor2, k, read_start, read_stride * 2, N / 2, even_saved, odd_saved);
            rN_odd  = fastFourierTransform(input, factors, new_factor2, k, read_start + read_stride, read_stride * 2, N / 2, even_saved, odd_saved) ;
        }
        rN = rN_even + rN_odd * factor2;
        even_saved = rN_even;
        odd_saved = rN_odd;
        return rN;
    }
}

complex<double> fastFourierTransform_radix3(
  const vector<complex<double>>& input,
  const vector<complex<double>>& factors,
  const complex<double>& factor2,
  size_t k, size_t read_start, size_t read_stride, size_t N,
  complex<double>& saved0, complex<double>& saved1, complex<double>& saved2) {
//   std::cout << "Main fft radix3 loop\n";
  complex<double> rN(0, 0);
  complex<double> rN0(0, 0);
  complex<double> rN1(0, 0);
  complex<double> rN2(0, 0);
  if (N % 3 != 0) {
//     if (N % 2 == 0) {
//       // TODO: radix-2 fft
//     }
    for (size_t j = 0; j < N; ++j) {
      const complex<double> tmp(0, -1.0 * j * 2.0 * M_PI * k / N);
      rN += input[read_start + j * read_stride]
          * std::exp(tmp);
    }
    return rN;
  } else {
    if (N == FFT_RADIX_3_MIN_N) {
      for (size_t j = 0; j < N / 3; ++j) {
        rN0 += input[read_start + 3 * j * read_stride] * factors[j];
        rN1 += input[read_start + (3 * j + 1) * read_stride] * factors[j];
        rN2 += input[read_start + (3 * j + 2) * read_stride] * factors[j];
      }
    } else {
      const complex<double> new_factor2 = factor2 * factor2 * factor2;
      const size_t new_stride = read_stride * 3;
      const size_t new_N = N / 3;
      rN0 = fastFourierTransform_radix3(input, factors, new_factor2, k, read_start, new_stride, new_N, saved0, saved1, saved2);
      rN1 = fastFourierTransform_radix3(input, factors, new_factor2, k, read_start + read_stride, new_stride, new_N, saved0, saved1, saved2);
      rN2 = fastFourierTransform_radix3(input, factors, new_factor2, k, read_start + read_stride * 2, new_stride, new_N, saved0, saved1, saved2);
    }
    rN = rN0 + rN1 * factor2 + rN2 * factor2 * factor2;
    saved0 = rN0;
    saved1 = rN1;
    saved2 = rN2;
    return rN;
  }
}

vector<complex<double>> fastFourierTransform(const vector<complex<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input.size();
    vector<complex<double>> result(N);
    vector<vector<complex<double>>> factors_table(N, vector<complex<double>>(FFT_MIN_N/2));
    for (size_t k = 0; k < N / 2; ++k) {
        complex<double> even(0, 0);
        complex<double> odd(0, 0);
        for (size_t i = 0; i < FFT_MIN_N/2; ++i) {
            factors_table[k][i] = std::exp(complex<double>(0, -1.0 * i * 2.0 * M_PI * k / (FFT_MIN_N / 2)));
        }
        const complex<double> factor2 = std::exp(complex<double>(0, -1.0 * 2.0 * M_PI * k / N));
        result[k] = fastFourierTransform(input, factors_table[k], factor2, k, 0, 1, N, even, odd);
        result[k + N / 2] = even - odd * factor2;
    }
    return result;
}

vector<complex<double>> fastFourierTransform_radix3(const vector<complex<double>>& input) {
//   std::cout << "Main fft radix entry\n";
  using namespace std::complex_literals;
  const size_t N = input.size();
  vector<complex<double>> result(N);
  vector<complex<double>> factors_table(FFT_RADIX_3_MIN_N/3);
  const complex<double> f23 = std::exp(complex<double>(0, -2.0 / 3.0 * M_PI));
  const complex<double> f43 = std::exp(complex<double>(0, -4.0 / 3.0 * M_PI));
  const complex<double> f83 = std::exp(complex<double>(0, -8.0 / 3.0 * M_PI));
  complex<double> r0(0, 0);
  complex<double> r1(0, 0);
  complex<double> r2(0, 0);
  for (size_t k = 0; k < N / 3; ++k) {
    for (size_t i = 0; i < FFT_RADIX_3_MIN_N/3; ++i) {
      factors_table[i] = std::exp(complex<double>(0, -1.0 * i * 2.0 * M_PI * k / (FFT_RADIX_3_MIN_N / 3)));
    }
    const complex<double> factor2 = std::exp(complex<double>(0, -1.0 * 2.0 * M_PI * k / N));
    result[k] = fastFourierTransform_radix3(input, factors_table, factor2, k, 0, 1, N, r0, r1, r2);
    result[k + N / 3] = r0 + r1 * factor2 * f23 + r2 * factor2 * factor2 * f43;
    result[k + 2 * N / 3] = r0 + r1 * factor2 * f43 + r2 * factor2 * factor2 * f83;
  }
  return result;
}

vector<complex<double>> fastFourierTransform(const vector<double>& input) {
    using namespace std::complex_literals;
    vector<complex<double>> complex_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        complex_input[i] = complex<double>(input[i], 0.0);
    }
    return fastFourierTransform(complex_input);
}

vector<complex<double>> fastFourierTransform_radix3(const vector<double>& input) {
    using namespace std::complex_literals;
    vector<complex<double>> complex_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        complex_input[i] = complex<double>(input[i], 0.0);
    }
    return fastFourierTransform_radix3(complex_input);
}

vector<complex<double>> reverseFourierTransform(const vector<complex<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input.size();
    vector<complex<double>> result(N);
    for (size_t i = 0; i < N; ++i) {
        complex<double> rN(0, 0);
        for (size_t j = 0; j < N; ++j) {
            const complex<double> tmp(0, 1.0 * j * 2.0 * M_PI * i / N);
            rN += input[j] * std::exp(tmp);
        }
        result[i] = rN;
    }
    return result;
}

complex<double> reverseFastFourierTransform(const vector<complex<double>>& input,
                                            const vector<complex<double>>& factors,
                                            const complex<double>& factor2,
                                            size_t k, size_t read_start, size_t read_stride, size_t N,
                                            complex<double>& even_saved, complex<double>& odd_saved) {
    using namespace std::complex_literals;
    complex<double> rN(0, 0);
    complex<double> rN_odd(0, 0);
    complex<double> rN_even(0, 0);
    if (N % 2 != 0) {
        for (size_t j = 0; j < N; ++j) {
            const complex<double> tmp(0, 1.0 * j * 2.0 * M_PI * k / N);
            rN += input[read_start + j * read_stride] * std::exp(tmp);
        }
        return rN;
    } else {
        if (N == FFT_MIN_N) {
            for (size_t j = 0; j < FFT_MIN_N / 2; ++j) {
                rN_even += input[read_start + 2 * j * read_stride] * factors[j];
                rN_odd  += input[read_start + (2 * j + 1) * read_stride] * factors[j];
            }
        } else {
            const complex<double> new_factor2 = factor2 * factor2;
            rN_even = reverseFastFourierTransform(input, factors, new_factor2, k, read_start, read_stride * 2, N / 2, even_saved, odd_saved);
            rN_odd  = reverseFastFourierTransform(input, factors, new_factor2, k, read_start + read_stride, read_stride * 2, N / 2, even_saved, odd_saved);
        }
        even_saved = rN_even;
        odd_saved = rN_odd;
        rN = rN_even + rN_odd * factor2;
        return rN;
    }
}

vector<complex<double>> reverseFastFourierTransform(const vector<complex<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input.size();
    vector<complex<double>> result(N);
    vector<vector<complex<double>>> factors_table(N, vector<complex<double>>(FFT_MIN_N/2));
    for (size_t k = 0; k < N / 2; ++k) {
        complex<double> even(0, 0);
        complex<double> odd(0, 0);
        for (size_t i = 0; i < FFT_MIN_N/2; ++i) {
            factors_table[k][i] = std::exp(complex<double>(0, 1.0 * i * 2.0 * M_PI * k / (FFT_MIN_N / 2)));
        }
        const complex<double> factor2 = std::exp(complex<double>(0, 1.0 * 2.0 * M_PI * k / N));
        result[k] = reverseFastFourierTransform(input, factors_table[k], factor2, k, 0, 1, N, even, odd);
        result[k + N / 2] = even - odd * factor2;
    }
    return result;
}

vector<complex<double>> fourierTransform(const vector<double>& input) {
    using namespace std::complex_literals;
    vector<complex<double>> complex_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        complex_input[i] = complex<double>(input[i], 0.0);
    }
    return fourierTransform(complex_input);
}

vector<complex<double>> fourierTransform(std::function<double(double)> f, double from, double to, double width = 0.001) {
    size_t bins = (size_t)std::ceil((to - from)/width);
    vector<double> input(bins);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = f(from + i * width);
    }
    return fourierTransform(input);
}

vector<vector<complex<double>>> fourierTransform2D(const vector<complex<double>>& inputX, const vector<complex<double>>& inputY) {
    using namespace std::complex_literals;
    const size_t N = inputX.size();
    const size_t M = inputY.size();
    vector<vector<complex<double>>> result(M, vector<complex<double>>(N));
    vector<complex<double>> resultX = fourierTransform(inputX);
    for (size_t j = 0; j < M; ++j) {
        complex<double> rM(0, 0);
        for (size_t jj = 0; jj < M; ++jj) {
            double tmp = -1.0 * jj * 2.0 * M_PI * j / M;
            rM += inputY[jj] * (std::cos(tmp) + complex<double>(0, std::sin(tmp)));
        }
        for (size_t i = 0; i < N; ++i) {
            resultX[i] *= rM;
        }
        result[j] = resultX;
    }
    return result;
}

// fourier transform 
vector<vector<complex<double>>> fourierTransform2D(const vector<vector<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input[0].size();
    const size_t M = input.size();
    vector<vector<complex<double>>> result(M, vector<complex<double>>(N));
    // transform along X direction
    for (size_t j = 0; j < M; ++j) {
        result[j] = fourierTransform(input[j]);
    }
    // transpose
    vector<vector<complex<double>>> tmp(N, vector<complex<double>>(M));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            tmp[i][j] = result[j][i];
        }
    }
    // transform along Y direction
    for (size_t i = 0; i < N; ++i) {
        tmp[i] = fourierTransform(tmp[i]);
    }
    // transpose back
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[j][i] = tmp[i][j];
        }
    }
    return result;
}

// fourier transform 
vector<vector<complex<double>>> fastFourierTransform2D(const vector<vector<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input[0].size();
    const size_t M = input.size();
    vector<vector<complex<double>>> result(M, vector<complex<double>>(N));
    // transform along X direction
    for (size_t j = 0; j < M; ++j) {
        result[j] = fastFourierTransform(input[j]);
    }
    // transpose
    vector<vector<complex<double>>> tmp(N, vector<complex<double>>(M));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            tmp[i][j] = result[j][i];
        }
    }
    // transform along Y direction
    for (size_t i = 0; i < N; ++i) {
        tmp[i] = fastFourierTransform(tmp[i]);
    }
    // transpose back
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[j][i] = tmp[i][j];
        }
    }
    return result;
}

vector<vector<complex<double>>> reverseFourierTransform2D(const vector<vector<complex<double>>>& input) {
    using namespace std::complex_literals;
    const size_t N = input[0].size();
    const size_t M = input.size();
    vector<vector<complex<double>>> result(M, vector<complex<double>>(N));
    // reverse transform along X direction
    for (size_t j = 0; j < M; ++j) {
        result[j] = reverseFourierTransform(input[j]);
    }
    // transpose
    vector<vector<complex<double>>> tmp(N, vector<complex<double>>(M));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            tmp[i][j] = result[j][i];
        }
    }
    // reverse transform along Y direction
    for (size_t i = 0; i < N; ++i) {
        tmp[i] = reverseFourierTransform(tmp[i]);
    }
    // reverse transpose back
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[j][i] = tmp[i][j];
        }
    }
    return result;
}

vector<vector<complex<double>>> reverseFastFourierTransform2D(const vector<vector<complex<double>>>& input) {
    using namespace std::complex_literals;
    const size_t N = input[0].size();
    const size_t M = input.size();
    vector<vector<complex<double>>> result(M, vector<complex<double>>(N));
    // reverse transform along X direction
    for (size_t j = 0; j < M; ++j) {
        result[j] = reverseFastFourierTransform(input[j]);
    }
    // transpose
    vector<vector<complex<double>>> tmp(N, vector<complex<double>>(M));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            tmp[i][j] = result[j][i];
        }
    }
    // reverse transform along Y direction
    for (size_t i = 0; i < N; ++i) {
        tmp[i] = reverseFastFourierTransform(tmp[i]);
    }
    // reverse transpose back
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[j][i] = tmp[i][j];
        }
    }
    return result;
}

void writeToFile(const vector<vector<complex<double>>>& matrix, const char* filename) {
    const std::string fn_real = std::string{filename} + ".real";
    const std::string fn_imag = std::string{filename} + ".imag";
    FILE* handle_real = fopen(fn_real.c_str(), "w");
    FILE* handle_imag = fopen(fn_imag.c_str(), "w");
    const size_t Y = matrix.size();
    const size_t X = matrix[0].size();
    for (size_t j = 0; j < Y; ++j) {
        for (size_t i = 0; i < X; ++i) {
            fprintf(handle_real, "%15.5f", matrix[j][i].real());
            fprintf(handle_imag, "%15.5f", matrix[j][i].imag());
        }
        fprintf(handle_real, "\n");
        fprintf(handle_imag, "\n");
    }
    fclose(handle_real);
    fclose(handle_imag);
}

void writeToFile(const vector<vector<double>>& matrix, const char* filename) {
    FILE* handle = fopen(filename, "w");
    const size_t Y = matrix.size();
    const size_t X = matrix[0].size();
    for (size_t j = 0; j < Y; ++j) {
        for (size_t i = 0; i < X; ++i) {
            fprintf(handle, "%15.5f", matrix[j][i]);
        }
        fprintf(handle, "\n");
    }
    fclose(handle);
} 
