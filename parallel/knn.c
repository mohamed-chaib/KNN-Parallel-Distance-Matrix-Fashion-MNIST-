#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define MAX_CLASSES 100

// Structure for sorting distances with indices
typedef struct {
    double distance;
    int index;
} DistanceIndex;

// Comparator for qsort
int compare(const void *a, const void *b) {
    double diff = ((DistanceIndex *)a)->distance - ((DistanceIndex *)b)->distance;
    return (diff > 0) - (diff < 0);
}

// Compute distances
void compute_distances(double *x, double *X_train, double *distances, int n_train, int dim) {
    for (int i = 0; i < n_train; i++) {
        double sum = 0.0;
        for (int j = 0; j < dim; j++) {
            double diff = X_train[i * dim + j] - x[j];
            sum += diff * diff;
        }
        distances[i] = sqrt(sum);
    }
}

// Predict one sample
int predict_one(double *x, double *X_train, int *y_train, int n_train, int dim, int k) {
    double *distances = (double *)malloc(n_train * sizeof(double));
    DistanceIndex *di = (DistanceIndex *)malloc(n_train * sizeof(DistanceIndex));

    compute_distances(x, X_train, distances, n_train, dim);
    for (int i = 0; i < n_train; i++) {
        di[i].distance = distances[i];
        di[i].index = i;
    }

    qsort(di, n_train, sizeof(DistanceIndex), compare);

    int counts[MAX_CLASSES] = {0};

    for (int i = 0; i < k; i++) {
        int label = y_train[di[i].index];
        counts[label]++;
    }

    int max_count = 0;
    int prediction = 0;

    for (int i = 0; i < MAX_CLASSES; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            prediction = i;
        }
    }

    free(distances);
    free(di);

    return prediction;
}

// Predict all samples

void predict_all_parallel(double *X_test, int n_test,
                 double *X_train, int *y_train,
                 int n_train, int dim, int k,
                 int *predictions) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_test; i++) {

        predictions[i] = predict_one(
            &X_test[i * dim],
            X_train,
            y_train,
            n_train,
            dim,
            k
        );

    }
}

void knn_set_num_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}

int knn_get_max_threads(void) {
    return omp_get_max_threads();
}

void knn_predict(double *X_test, int n_test,
                 double *X_train, int *y_train,
                 int n_train, int dim, int k,
                 int *predictions) {

    predict_all_parallel(
        X_test, n_test,
        X_train, y_train,
        n_train, dim, k,
        predictions
    );
}
