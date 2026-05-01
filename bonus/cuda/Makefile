NVCC = nvcc
CFLAGS = -O3 -Xcompiler -fPIC -shared

libknn_cuda.so: knn_cuda.cu
	$(NVCC) $(CFLAGS) -o libknn_cuda.so knn_cuda.cu

clean:
	rm -f libknn_cuda.so
