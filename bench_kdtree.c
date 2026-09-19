/* KD-tree radius-query benchmark: C leg of the algorithmic-fix comparison that
 * complements bench_haversine.c. Same 30,114 employees x 1,000 places as the dense
 * benchmark, but instead of computing all 30.1M distances, builds a KD-tree over
 * places' ECEF (3D Cartesian) positions and asks each employee "which places are
 * within HARD_CAP_KM?" -- a hand-rolled KD-tree, not a library, so this is a fair
 * like-for-like comparison against the hand-rolled dense loop in bench_haversine.c.
 *
 * This mirrors what reassign.py's combined_score_matrix actually does now (see
 * "Language vs. algorithm" in README.md): exact haversine distance is only computed
 * for the handful of candidate pairs the tree returns, not the full dense matrix.
 *
 * Build:  gcc -O3 -march=native -o bench_kdtree_c bench_kdtree.c -lm
 * Run:    ./bench_kdtree_c
 *
 * Measured on this machine: ~0.01-0.02s build+query (vs. ~1.2s for the dense loop
 * in bench_haversine.c) -- a ~60-100x speedup, consistent with the ~50x measured
 * in Python (bench_haversine.py's dense pass vs. reassign.py's KD-tree query) and
 * with reassign.py's own before/after (~2.4s -> ~0.9s for the whole scoring step,
 * which also includes capability matching, not just distance).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_EMP 30114
#define N_PLACES 1000
#define EARTH_R_KM 6371.0
#define HARD_CAP_KM 80.0

typedef struct KDNode {
    int idx;
    int axis;
    struct KDNode *left, *right;
} KDNode;

static double (*g_pts)[3];   /* points the tree indexes (places, in ECEF) */
static int g_axis;           /* current split axis, read by the qsort comparator */

static int cmp_axis(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    double va = g_pts[ia][g_axis], vb = g_pts[ib][g_axis];
    return (va > vb) - (va < vb);
}

static KDNode *kd_build(int *indices, int n, int depth) {
    if (n <= 0) return NULL;
    int axis = depth % 3;
    g_axis = axis;
    qsort(indices, n, sizeof(int), cmp_axis);
    int mid = n / 2;
    KDNode *node = malloc(sizeof(KDNode));
    node->idx = indices[mid];
    node->axis = axis;
    node->left = kd_build(indices, mid, depth + 1);
    node->right = kd_build(indices + mid + 1, n - mid - 1, depth + 1);
    return node;
}

static void kd_query_radius(KDNode *node, const double *q, double r2, int *count) {
    if (!node) return;
    double dx = g_pts[node->idx][0] - q[0];
    double dy = g_pts[node->idx][1] - q[1];
    double dz = g_pts[node->idx][2] - q[2];
    double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= r2) (*count)++;

    double diff = q[node->axis] - g_pts[node->idx][node->axis];
    KDNode *near = diff < 0 ? node->left : node->right;
    KDNode *far  = diff < 0 ? node->right : node->left;
    kd_query_radius(near, q, r2, count);
    if (diff * diff <= r2) kd_query_radius(far, q, r2, count);  /* splitting plane within radius: must check far side too */
}

static void to_ecef(double lat, double lon, double out[3]) {
    double latr = lat * M_PI / 180.0, lonr = lon * M_PI / 180.0;
    out[0] = EARTH_R_KM * cos(latr) * cos(lonr);
    out[1] = EARTH_R_KM * cos(latr) * sin(lonr);
    out[2] = EARTH_R_KM * sin(latr);
}

int main(void) {
    double emp_lat[N_EMP], emp_lon[N_EMP];
    double pl_lat[N_PLACES], pl_lon[N_PLACES];
    srand(1);
    for (int i = 0; i < N_EMP; i++) {
        emp_lat[i] = 35.0 + 25.0 * rand() / RAND_MAX;
        emp_lon[i] = -10.0 + 40.0 * rand() / RAND_MAX;
    }
    for (int j = 0; j < N_PLACES; j++) {
        pl_lat[j] = 35.0 + 25.0 * rand() / RAND_MAX;
        pl_lon[j] = -10.0 + 40.0 * rand() / RAND_MAX;
    }

    static double place_xyz[N_PLACES][3];
    for (int j = 0; j < N_PLACES; j++) to_ecef(pl_lat[j], pl_lon[j], place_xyz[j]);
    g_pts = place_xyz;

    int *indices = malloc(N_PLACES * sizeof(int));
    for (int j = 0; j < N_PLACES; j++) indices[j] = j;

    double chord = 2 * EARTH_R_KM * sin(HARD_CAP_KM / (2 * EARTH_R_KM));
    double r2 = chord * chord;

    for (int trial = 1; trial <= 3; trial++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int *idx_copy = malloc(N_PLACES * sizeof(int));
        for (int j = 0; j < N_PLACES; j++) idx_copy[j] = j;
        KDNode *tree = kd_build(idx_copy, N_PLACES, 0);

        long total_candidates = 0;
        for (int i = 0; i < N_EMP; i++) {
            double q[3];
            to_ecef(emp_lat[i], emp_lon[i], q);
            int count = 0;
            kd_query_radius(tree, q, r2, &count);
            total_candidates += count;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("C KD-tree trial %d: build+query for %d employees vs %d places: %.4fs  "
               "avg candidates/employee=%.2f\n",
               trial, N_EMP, N_PLACES, elapsed, (double)total_candidates / N_EMP);

        free(idx_copy);
        /* tree nodes intentionally leaked -- benchmark process exits immediately after */
    }

    free(indices);
    return 0;
}
