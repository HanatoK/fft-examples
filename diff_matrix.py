#!/usr/bin/env python3
import numpy as np

def diff_matrices(input_file1, input_file2, output_file):
    data1 = np.genfromtxt(input_file1, unpack = True)
    data2 = np.genfromtxt(input_file2, unpack = True)
    diff = data1 - data2
    np.savetxt(output_file, diff, fmt='%15.5f')
    abs_diff = np.abs(diff)
    total_error = np.sum(abs_diff)
    print(f'Error between {input_file1} and {input_file2} is {total_error:12.5f}')

diff_matrices('cufft_output1_r2c.dat.real', 'fftw_output1_r2c.dat.real', '1.real.err')
diff_matrices('cufft_output2_r2c.dat.real', 'fftw_output2_r2c.dat.real', '2.real.err')
diff_matrices('cufft_output1_r2c.dat.imag', 'fftw_output1_r2c.dat.imag', '1.imag.err')
diff_matrices('cufft_output2_r2c.dat.imag', 'fftw_output2_r2c.dat.imag', '2.imag.err')

diff_matrices('cufft_output1_c2r.dat', 'fftw_output1_c2r.dat', '1.c2r.err')
diff_matrices('cufft_output2_c2r.dat', 'fftw_output2_c2r.dat', '2.c2r.err')
