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

vector<complex<double>> fourierTransform(const vector<complex<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input.size();
    const double PI = std::acos(-1.0);
    vector<complex<double>> result(N);
    for (size_t k = 0; k < N; ++k) {
        complex<double> rN(0, 0);
        for (size_t j = 0; j < N; ++j) {
            const double tmp = -1.0 * j * 2.0 * PI * k / N;
            rN += input[j] * (std::cos(tmp) + complex<double>(0, std::sin(tmp)));
        }
        result[k] = rN;
    }
    return result;
}

vector<complex<double>> reverseFourierTransform(const vector<complex<double>>& input) {
    using namespace std::complex_literals;
    const size_t N = input.size();
    const double PI = std::acos(-1.0);
    vector<complex<double>> result(N);
    for (size_t i = 0; i < N; ++i) {
        complex<double> rN(0, 0);
        for (size_t j = 0; j < N; ++j) {
            double tmp = 1.0 * j * 2.0 * PI * i / N;
            rN += input[j] * (std::cos(tmp) + complex<double>(0, std::sin(tmp)));
        }
//         result[i] = rN / complex<double>(double(N), 0);
        result[i] = rN;
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
    const double PI = std::acos(-1.0);
    vector<vector<complex<double>>> result(M, vector<complex<double>>(N));
    vector<complex<double>> resultX = fourierTransform(inputX);
    for (size_t j = 0; j < M; ++j) {
        complex<double> rM(0, 0);
        for (size_t jj = 0; jj < M; ++jj) {
            double tmp = -1.0 * jj * 2.0 * PI * j / M;
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
