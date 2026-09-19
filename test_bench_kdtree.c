#include <assert.h>
#include <stdio.h>
#include <math.h>

#define main bench_main
#include "bench_kdtree.c"
#undef main

void test_to_ecef() {
    double out[3];
    to_ecef(0.0, 0.0, out);
    assert(fabs(out[0] - EARTH_R_KM) < 1e-6);
    assert(fabs(out[1] - 0.0) < 1e-6);
    assert(fabs(out[2] - 0.0) < 1e-6);

    to_ecef(90.0, 0.0, out);
    assert(fabs(out[0] - 0.0) < 1e-6);
    assert(fabs(out[1] - 0.0) < 1e-6);
    assert(fabs(out[2] - EARTH_R_KM) < 1e-6);
}

void test_kd_tree() {
    double test_pts[3][3] = {
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {0.0, 10.0, 0.0}
    };
    g_pts = test_pts;
    int indices[] = {0, 1, 2};
    KDNode *tree = kd_build(indices, 3, 0);
    assert(tree != NULL);
    double q1[3] = {0.0, 0.0, 0.0};
    int count = 0;
    kd_query_radius(tree, q1, 25.0, &count);
    assert(count == 1);
    double q2[3] = {0.0, 0.0, 0.0};
    count = 0;
    kd_query_radius(tree, q2, 225.0, &count);
    assert(count == 3);
    
    // Explicitly test cmp_axis to ensure branch coverage if needed
    g_axis = 0; // compare X
    int a = 0, b = 1; // 0.0 vs 10.0
    assert(cmp_axis(&a, &b) < 0);
    assert(cmp_axis(&b, &a) > 0);
    assert(cmp_axis(&a, &a) == 0);

    free(tree->left);
    free(tree->right);
    free(tree);
}

int main() {
    test_to_ecef();
    test_kd_tree();
    // Run the integration test / main loop to get 100% coverage
    bench_main();
    printf("All KD-tree C tests passed!\n");
    return 0;
}
