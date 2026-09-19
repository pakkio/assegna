#include <assert.h>
#define main match_main
#include "match.c"
#undef main

void test_haversine() {
    double d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437);
    assert(d > 3900 && d < 4000);
}

void test_overlap() {
    User u;
    u.num_caps = 2;
    strcpy(u.caps[0], "python");
    strcpy(u.caps[1], "c");

    Place p;
    p.num_req_caps = 1;
    strcpy(p.req_caps[0], "python");

    double o = overlap_score(&u, &p);
    assert(o > 0.99 && o < 1.01);

    p.num_req_caps = 0;
    assert(overlap_score(&u, &p) == 0.0);
}

void test_mcmf() {
    init_graph(4);
    add_edge(0, 1, 1, 0.0);
    add_edge(1, 2, 1, -10.0);
    add_edge(2, 3, 1, 0.0);
    
    solve_mcmf(0, 3, 4, false, 1);
    
    int flow = 0;
    for (int i = 0; i < edge_count[0]; i++) flow += graph[0][i].flow;
    assert(flow == 1);
    
    free_graph(4);
}

void test_llm_score() {
    setenv("OPENCODEGO_API_KEY", "dummy", 1);
    User u; u.num_caps = 1; strcpy(u.caps[0], "x");
    Place p; strcpy(p.name, "Y"); p.num_req_caps = 1; strcpy(p.req_caps[0], "y");
    double s = llm_score(&u, &p);
}

void test_dotenv() {
    system("echo 'OPENCODEGO_API_KEY=dummy' > test.env");
    load_dotenv("test.env");
    load_dotenv("nonexistent.env");
}

int main() {
    printf("Testing match.c...\n");
    test_haversine();
    test_overlap();
    test_mcmf();
    test_llm_score();
    test_dotenv();
    
    char* argv[] = {"match", "--allow-unassigned", "--llm", "--users", "../users.json", "--places", "../places.json"};
    match_main(7, argv);
    
    printf("All match.c tests passed!\n");
    return 0;
}
