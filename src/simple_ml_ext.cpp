#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void mat_mul(const float *A, const float *B, float *C, size_t m, size_t k, size_t n)
{
  for(size_t i = 0; i < m; i++){
    for(size_t j = 0; j < n; j++){
      C[i * n + j] = 0;
      for(size_t p = 0; p < k; p++){
        C[i * n + j] += A[i * k + p] * B[p * n + j];
      }
    }
  }
}
void transpose(const float *A, float *B, size_t m, size_t n)
{
  for(size_t i = 0; i < m; i++){
    for(size_t j = 0; j < n; j++){
      B[j * m + i] = A[i * n + j];
    }
  }
}
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_examples = m;
    size_t epoch = (num_examples + batch - 1) / batch;
    for(size_t i = 0; i < epoch; i++){
      int start = i * batch;
      const float *X_batch = &X[start * n];
      float *Z = new float[batch * k];
      mat_mul(X_batch, theta, Z, batch, n, k);
      for(size_t i = 0; i < batch * k; i++){
        Z[i] = expf(Z[i]);
      }
      for(size_t i = 0; i < batch; i++){
        float sum = 0;
        for(size_t j = 0; j < k; j++){
          sum += Z[i * k + j];
        }
        for(size_t j = 0; j < k; j++){
          Z[i * k + j] /= sum;
          if(j == y[start + i]){
            Z[i * k + j]--;
          }
        }
      }
      float *X_T = new float[n * batch];
      transpose(X_batch, X_T, batch, n);
      float *grad = new float[n * k];
      mat_mul(X_T, Z, grad, n, batch, k);
      for(size_t i = 0; i < n * k; i++){
        theta[i] -= lr * grad[i] / (float)batch;
      }
      delete[] Z;
      delete[] X_T;
      delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
