#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <ctype.h>
#include <cJSON.h>
#include <curl/curl.h>

#define MAX_USERS 1000
#define MAX_PLACES 1000
#define MAX_CAPS 50

#define MAX_RELOCATION_KM 30.0
#define OVER_CAP_PENALTY_PER_KM 0.005
#define EARTH_RADIUS_KM 6371.0
#define LLM_THRESHOLD 0.5

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    char name[100];
    char caps[MAX_CAPS][100];
    int num_caps;
    double lat, lon;
} User;

typedef struct {
    char name[100];
    char req_caps[MAX_CAPS][100];
    int num_req_caps;
    double lat, lon;
    int capacity;
} Place;

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
    double p1 = lat1 * M_PI / 180.0;
    double p2 = lat2 * M_PI / 180.0;
    double dphi = (lat2 - lat1) * M_PI / 180.0;
    double dlambda = (lon2 - lon1) * M_PI / 180.0;
    double a = sin(dphi / 2.0) * sin(dphi / 2.0) +
               cos(p1) * cos(p2) * sin(dlambda / 2.0) * sin(dlambda / 2.0);
    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(a));
}

double overlap_score(User* u, Place* p) {
    if (p->num_req_caps == 0) return 0.0;
    int match = 0;
    for (int i = 0; i < p->num_req_caps; i++) {
        for (int j = 0; j < u->num_caps; j++) {
            if (strcasecmp(p->req_caps[i], u->caps[j]) == 0) {
                match++;
                break;
            }
        }
    }
    return (double)match / p->num_req_caps;
}

struct MemoryStruct {
    char *memory;
    size_t size;
};

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if(!ptr) return 0;
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    return realsize;
}

double llm_score(User* u, Place* p) {
    const char* api_key = getenv("OPENCODEGO_API_KEY");
    if (!api_key) {
        fprintf(stderr, "OPENCODEGO_API_KEY not found in environment\n");
        exit(1);
    }
    const char* model = getenv("OPENCODEGO_MODEL");
    if (!model) model = "deepseek-v4-flash";
    const char* base_url = "https://opencode.ai/zen/go/v1/chat/completions";

    char prompt[2048] = "User capabilities: ";
    for (int i = 0; i < u->num_caps; i++) {
        strcat(prompt, u->caps[i]);
        if (i < u->num_caps - 1) strcat(prompt, ", ");
    }
    strcat(prompt, "\nPlace '");
    strcat(prompt, p->name);
    strcat(prompt, "' required capabilities: ");
    for (int i = 0; i < p->num_req_caps; i++) {
        strcat(prompt, p->req_caps[i]);
        if (i < p->num_req_caps - 1) strcat(prompt, ", ");
    }
    strcat(prompt, "\nRate how well this user semantically fits the place's requirements, from 0.0 (no fit) to 1.0 (perfect fit), giving partial credit for related or adjacent skills even without an exact string match. Reply with ONLY a number between 0.0 and 1.0.");

    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model", model);
    cJSON* messages = cJSON_AddArrayToObject(root, "messages");
    cJSON* msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "role", "user");
    cJSON_AddStringToObject(msg, "content", prompt);
    cJSON_AddItemToArray(messages, msg);
    cJSON_AddNumberToObject(root, "temperature", 0.0);

    char* json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    chunk.memory = malloc(1);
    chunk.size = 0;
    double score = 0.0;

    curl = curl_easy_init();
    if(curl) {
        struct curl_slist *headers = NULL;
        char auth_header[256];
        snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
        headers = curl_slist_append(headers, auth_header);
        headers = curl_slist_append(headers, "Content-Type: application/json");
        // Using a static uuid for simplicity as in original code
        headers = curl_slist_append(headers, "x-opencode-session: 12345678-1234-1234-1234-123456789012");

        curl_easy_setopt(curl, CURLOPT_URL, base_url);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);

        res = curl_easy_perform(curl);
        if(res == CURLE_OK) {
            cJSON* resp = cJSON_Parse(chunk.memory);
            if(resp) {
                cJSON* choices = cJSON_GetObjectItem(resp, "choices");
                if (cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
                    cJSON* choice = cJSON_GetArrayItem(choices, 0);
                    cJSON* message = cJSON_GetObjectItem(choice, "message");
                    cJSON* content = cJSON_GetObjectItem(message, "content");
                    if (cJSON_IsString(content)) {
                        score = atof(content->valuestring);
                        if (score < 0.0) score = 0.0;
                        if (score > 1.0) score = 1.0;
                    }
                }
                cJSON_Delete(resp);
            }
        }
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    free(json_str);
    free(chunk.memory);
    return score;
}

void load_dotenv(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char* sep = strchr(line, '=');
        if (sep) {
            *sep = 0;
            char* val = sep + 1;
            char* newline = strchr(val, '\n');
            if (newline) *newline = 0;
            setenv(line, val, 1);
        }
    }
    fclose(f);
}

// Graph for custom min-cost max-flow
#define MAX_NODES (MAX_USERS + MAX_PLACES + 2)
typedef struct {
    int to;
    int cap;
    int flow;
    double cost;
    int rev;
} Edge;

Edge* graph[MAX_NODES];
int edge_count[MAX_NODES];
int edge_capacity[MAX_NODES];

void init_graph(int n) {
    for (int i = 0; i < n; i++) {
        edge_count[i] = 0;
        edge_capacity[i] = 16;
        graph[i] = malloc(sizeof(Edge) * edge_capacity[i]);
    }
}

void free_graph(int n) {
    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
}

void add_edge(int from, int to, int cap, double cost) {
    if (edge_count[from] >= edge_capacity[from]) {
        edge_capacity[from] *= 2;
        graph[from] = realloc(graph[from], sizeof(Edge) * edge_capacity[from]);
    }
    if (edge_count[to] >= edge_capacity[to]) {
        edge_capacity[to] *= 2;
        graph[to] = realloc(graph[to], sizeof(Edge) * edge_capacity[to]);
    }
    int from_idx = edge_count[from]++;
    int to_idx = edge_count[to]++;
    graph[from][from_idx] = (Edge){to, cap, 0, cost, to_idx};
    graph[to][to_idx] = (Edge){from, 0, 0, -cost, from_idx};
}

double dist[MAX_NODES];
int parent_node[MAX_NODES];
int parent_edge[MAX_NODES];
bool in_queue[MAX_NODES];
int queue[MAX_NODES * 10];

bool spfa(int s, int t, int n) {
    for (int i = 0; i < n; i++) {
        dist[i] = 1e9;
        parent_node[i] = -1;
        in_queue[i] = false;
    }
    int head = 0, tail = 0;
    queue[tail++] = s;
    dist[s] = 0;
    in_queue[s] = true;
    
    while (head < tail) {
        int u = queue[head++];
        in_queue[u] = false;
        
        for (int i = 0; i < edge_count[u]; i++) {
            Edge* e = &graph[u][i];
            if (e->cap - e->flow > 0 && dist[e->to] > dist[u] + e->cost + 1e-7) {
                dist[e->to] = dist[u] + e->cost;
                parent_node[e->to] = u;
                parent_edge[e->to] = i;
                
                if (!in_queue[e->to]) {
                    queue[tail++] = e->to;
                    in_queue[e->to] = true;
                }
            }
        }
    }
    return dist[t] != 1e9;
}

void solve_mcmf(int s, int t, int n, bool want_full, int total_users) {
    int flow = 0;
    while (spfa(s, t, n)) {
        if (!want_full && dist[t] > -1e-7) break;
        if (want_full && flow == total_users) break;
        
        int push = 1e9;
        int curr = t;
        while (curr != s) {
            int p = parent_node[curr];
            int p_e = parent_edge[curr];
            int avail = graph[p][p_e].cap - graph[p][p_e].flow;
            if (avail < push) push = avail;
            curr = p;
        }
        curr = t;
        while (curr != s) {
            int p = parent_node[curr];
            int p_e = parent_edge[curr];
            int rev = graph[p][p_e].rev;
            graph[p][p_e].flow += push;
            graph[curr][rev].flow -= push;
            curr = p;
        }
        flow += push;
    }
}

typedef struct {
    char place[100];
    char user[100];
    double capability_score;
    double distance_km;
    char note[512];
} Result;

int cmp_results(const void* a, const void* b) {
    Result* r1 = (Result*)a;
    Result* r2 = (Result*)b;
    int cmp = strcmp(r1->place, r2->place);
    if (cmp != 0) return cmp;
    if (r1->capability_score > r2->capability_score) return -1;
    if (r1->capability_score < r2->capability_score) return 1;
    return 0;
}

void read_json_file(const char* filename, char** buffer) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    *buffer = malloc(length + 1);
    fread(*buffer, 1, length, f);
    (*buffer)[length] = '\0';
    fclose(f);
}

int main(int argc, char** argv) {
    const char* users_file = "users.json";
    const char* places_file = "places.json";
    bool use_llm = false;
    bool allow_unassigned = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--users") == 0 && i + 1 < argc) users_file = argv[++i];
        else if (strcmp(argv[i], "--places") == 0 && i + 1 < argc) places_file = argv[++i];
        else if (strcmp(argv[i], "--llm") == 0) use_llm = true;
        else if (strcmp(argv[i], "--allow-unassigned") == 0) allow_unassigned = true;
    }

    load_dotenv("../.env");
    curl_global_init(CURL_GLOBAL_ALL);

    char *users_str, *places_str;
    read_json_file(users_file, &users_str);
    read_json_file(places_file, &places_str);

    cJSON *u_json = cJSON_Parse(users_str);
    if (!u_json) { printf("Error parsing users: %s\n", cJSON_GetErrorPtr()); return 1; }
    cJSON *p_json = cJSON_Parse(places_str);
    if (!p_json) { printf("Error parsing places: %s\n", cJSON_GetErrorPtr()); return 1; }

    static User users[MAX_USERS];
    int num_users = 0;
    cJSON *item = NULL;
    cJSON_ArrayForEach(item, u_json) {
        strcpy(users[num_users].name, cJSON_GetObjectItem(item, "name")->valuestring);
        users[num_users].lat = cJSON_GetObjectItem(item, "lat")->valuedouble;
        users[num_users].lon = cJSON_GetObjectItem(item, "lon")->valuedouble;
        cJSON* caps = cJSON_GetObjectItem(item, "capabilities");
        int c_count = 0;
        cJSON* c = NULL;
        cJSON_ArrayForEach(c, caps) {
            strcpy(users[num_users].caps[c_count++], c->valuestring);
        }
        users[num_users].num_caps = c_count;
        num_users++;
    }

    static Place places[MAX_PLACES];
    int num_places = 0;
    int total_capacity = 0;
    cJSON_ArrayForEach(item, p_json) {
        strcpy(places[num_places].name, cJSON_GetObjectItem(item, "name")->valuestring);
        places[num_places].lat = cJSON_GetObjectItem(item, "lat")->valuedouble;
        places[num_places].lon = cJSON_GetObjectItem(item, "lon")->valuedouble;
        cJSON* cap = cJSON_GetObjectItem(item, "capacity");
        places[num_places].capacity = cap ? cap->valueint : 1;
        total_capacity += places[num_places].capacity;
        
        cJSON* caps = cJSON_GetObjectItem(item, "required_capabilities");
        int c_count = 0;
        cJSON* c = NULL;
        cJSON_ArrayForEach(c, caps) {
            strcpy(places[num_places].req_caps[c_count++], c->valuestring);
        }
        places[num_places].num_req_caps = c_count;
        num_places++;
    }

    double* cap_scores = malloc(num_users * num_places * sizeof(double));
    double* distances = malloc(num_users * num_places * sizeof(double));

    for (int i = 0; i < num_users; i++) {
        for (int j = 0; j < num_places; j++) {
            distances[i * num_places + j] = haversine_km(users[i].lat, users[i].lon, places[j].lat, places[j].lon);
            double base = overlap_score(&users[i], &places[j]);
            if (use_llm && base < LLM_THRESHOLD) {
                double l_score = llm_score(&users[i], &places[j]);
                if (l_score > base) base = l_score;
            }
            cap_scores[i * num_places + j] = base;
        }
    }

    bool want_full = !allow_unassigned;
    if (want_full && total_capacity < num_users) {
        printf("(total capacity %d < %d users -- can't force full assignment, falling back to best-available)\n\n", total_capacity, num_users);
        want_full = false;
    }

    int source = num_users + num_places;
    int sink = num_users + num_places + 1;
    int total_nodes = num_users + num_places + 2;
    init_graph(total_nodes);

    for (int i = 0; i < num_users; i++) {
        add_edge(source, i, 1, 0.0);
        for (int j = 0; j < num_places; j++) {
            double dist = distances[i * num_places + j];
            double over_cap = dist > MAX_RELOCATION_KM ? dist - MAX_RELOCATION_KM : 0.0;
            double combined = cap_scores[i * num_places + j] - over_cap * OVER_CAP_PENALTY_PER_KM;
            add_edge(i, num_users + j, 1, -combined);
        }
    }
    for (int j = 0; j < num_places; j++) {
        add_edge(num_users + j, sink, places[j].capacity, 0.0);
    }

    solve_mcmf(source, sink, total_nodes, want_full, num_users);

    Result results[MAX_USERS];
    int num_results = 0;

    for (int i = 0; i < num_users; i++) {
        for (int e = 0; e < edge_count[i]; e++) {
            Edge* edge = &graph[i][e];
            if (edge->to >= num_users && edge->to < num_users + num_places && edge->flow == 1) {
                int j = edge->to - num_users;
                Result r;
                strcpy(r.place, places[j].name);
                strcpy(r.user, users[i].name);
                r.capability_score = cap_scores[i * num_places + j];
                r.distance_km = distances[i * num_places + j];
                r.note[0] = '\0';
                
                if (r.distance_km > MAX_RELOCATION_KM) {
                    int best_i = -1;
                    double best_score = -1;
                    for (int k = 0; k < num_users; k++) {
                        if (k != i && distances[k * num_places + j] <= MAX_RELOCATION_KM) {
                            if (cap_scores[k * num_places + j] > best_score) {
                                best_score = cap_scores[k * num_places + j];
                                best_i = k;
                            }
                        }
                    }
                    if (best_i == -1) {
                        sprintf(r.note, "No candidate lives within %.0fkm of %s; %s is the closest available fit at %.1fkm, so the cap was exceeded.", MAX_RELOCATION_KM, places[j].name, users[i].name, r.distance_km);
                    } else {
                        sprintf(r.note, "%s relocates %.1fkm (exceeds the %.0fkm cap) because their capability fit (%.2f) clearly beats the best in-range candidate, %s (fit %.2f) within %.0fkm.", users[i].name, r.distance_km, MAX_RELOCATION_KM, r.capability_score, users[best_i].name, best_score, MAX_RELOCATION_KM);
                    }
                }
                results[num_results++] = r;
            }
        }
    }

    qsort(results, num_results, sizeof(Result), cmp_results);

    int unmatched = num_users - num_results;
    if (unmatched > 0) {
        printf("(%d user(s) left unassigned -- no place had net-positive value for them)\n\n", unmatched);
    }

    for (int i = 0; i < num_results; i++) {
        printf("%-30s -> %-20s (fit=%.3f, dist=%.1fkm)\n", results[i].place, results[i].user, results[i].capability_score, results[i].distance_km);
        if (strlen(results[i].note) > 0) {
            printf("  note: %s\n", results[i].note);
        }
    }

    free(cap_scores);
    free(distances);
    free_graph(total_nodes);
    cJSON_Delete(u_json);
    cJSON_Delete(p_json);
    free(users_str);
    free(places_str);
    curl_global_cleanup();

    return 0;
}
