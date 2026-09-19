/* Dense haversine benchmark: C leg of a 3-language (C / Scala / Python) comparison
 * for reassign.py's actual bottleneck -- combined_score_matrix's distance computation
 * over every (employee, place) pair, before top-K / HARD_CAP_KM pruning.
 *
 * Same N_EMP x N_PLACES = 30,114 x 1,000 = 30.1M pairs as the Python and Scala
 * versions, same formula, independently seeded RNG (not meant to produce a matching
 * checksum -- this measures wall-clock, not numerical agreement).
 *
 * Build:  gcc -O3 -march=native -o bench_haversine_c bench_haversine.c -lm
 * Run:    ./bench_haversine_c
 *
 * Measured on this machine: ~1.2s (vs. ~1.75s in Python/numpy, ~1.2-1.4s in
 * Scala/JVM -- see bench_haversine.py and BenchHaversine.scala). C and Scala come
 * out roughly tied here; both beat numpy by a modest ~1.5x, not an order of
 * magnitude -- all three call the same underlying libm sin/cos/asin. The real fix
 * for this bottleneck is a spatial index (~50x, see reassign.py's KD-tree
 * discussion), which helps equally in all three languages -- that gain dwarfs any
 * language choice made here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_EMP 30114
#define N_PLACES 1000
#define EARTH_R_KM 6371.0

int main(void) {
    double *p_lat = malloc(N_EMP * sizeof(double));
    double *p_lon = malloc(N_EMP * sizeof(double));
    double *pl_lat = malloc(N_PLACES * sizeof(double));
    double *pl_lon = malloc(N_PLACES * sizeof(double));
    double *dist = malloc((long)N_EMP * N_PLACES * sizeof(double));

    srand(1);
    for (int i = 0; i < N_EMP; i++) {
        p_lat[i] = 35.0 + 25.0 * rand() / RAND_MAX;
        p_lon[i] = -10.0 + 40.0 * rand() / RAND_MAX;
    }
    for (int j = 0; j < N_PLACES; j++) {
        pl_lat[j] = 35.0 + 25.0 * rand() / RAND_MAX;
        pl_lon[j] = -10.0 + 40.0 * rand() / RAND_MAX;
    }

    for (int trial = 1; trial <= 3; trial++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int i = 0; i < N_EMP; i++) {
            double lat1 = p_lat[i] * M_PI / 180.0;
            for (int j = 0; j < N_PLACES; j++) {
                double lat2 = pl_lat[j] * M_PI / 180.0;
                double dphi = (pl_lat[j] - p_lat[i]) * M_PI / 180.0;
                double dlambda = (pl_lon[j] - p_lon[i]) * M_PI / 180.0;
                double a = sin(dphi / 2) * sin(dphi / 2)
                         + cos(lat1) * cos(lat2) * sin(dlambda / 2) * sin(dlambda / 2);
                dist[(long)i * N_PLACES + j] = 2 * EARTH_R_KM * asin(sqrt(a));
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("C dense haversine trial %d (%d x %d = %ld pairs): %.3fs  checksum=%f\n",
               trial, N_EMP, N_PLACES, (long)N_EMP * N_PLACES, elapsed, dist[12345]);
    }

    free(p_lat); free(p_lon); free(pl_lat); free(pl_lon); free(dist);
    return 0;
}
