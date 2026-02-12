#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int n_embd;
    int n_head;
    int n_layer;
    int block_size;
    int head_dim;
    int vocab_size;
} Config;

typedef struct {
    float* wte;      // [V, E]
    float* wpe;      // [B, E]
    float* lm_head;  // [V, E]

    float* attn_wq;  // [L, E, E]
    float* attn_wk;  // [L, E, E]
    float* attn_wv;  // [L, E, E]
    float* attn_wo;  // [L, E, E]

    float* mlp_fc1;  // [L, 4E, E]
    float* mlp_fc2;  // [L, E, 4E]
} Weights;

typedef struct {
    float* wte;
    float* wpe;
    float* lm_head;

    float* attn_wq;
    float* attn_wk;
    float* attn_wv;
    float* attn_wo;

    float* mlp_fc1;
    float* mlp_fc2;
} Grads;

typedef struct {
    float* wte;
    float* wpe;
    float* lm_head;

    float* attn_wq;
    float* attn_wk;
    float* attn_wv;
    float* attn_wo;

    float* mlp_fc1;
    float* mlp_fc2;
} AdamBuf;

typedef struct {
    int n;            // sequence length
    int* in_tokens;   // [n]
    int* targets;     // [n]

    float* embed_sum;     // [n, E]
    float* x0;            // [n, E]
    float* x_final;       // [n, E]
    float* logits;        // [n, V]

    float* x_in;          // [L, n, E]
    float* xn_attn;       // [L, n, E]
    float* q;             // [L, n, E]
    float* k;             // [L, n, E]
    float* v;             // [L, n, E]
    float* attn_out;      // [L, n, E]
    float* x_after_attn;  // [L, n, E]
    float* xn_mlp;        // [L, n, E]
    float* h1;            // [L, n, 4E]
    float* h2;            // [L, n, 4E]
    float* x_out;         // [L, n, E]

    float* attn_probs;    // [L, n, H, n]
} TrainCache;

static float rand_uniform(void) {
    return (float)rand() / (float)RAND_MAX;
}

static float randn(float std) {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    float r = sqrtf(-2.0f * logf(fmaxf(u1, 1e-7f)));
    float theta = 2.0f * (float)M_PI * u2;
    return std * r * cosf(theta);
}

static void init_mat(float* w, size_t n, float std) {
    for (size_t i = 0; i < n; i++) w[i] = randn(std);
}

static void zero_mat(float* w, size_t n) {
    memset(w, 0, n * sizeof(float));
}

static bool load_lines(const char* path, char*** out_lines, int* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return false;
    }
    fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[sz] = '\0';

    int cap = 1024;
    int n = 0;
    char** lines = (char**)malloc((size_t)cap * sizeof(char*));
    if (!lines) {
        free(buf);
        return false;
    }

    char* save = NULL;
    char* tok = strtok_r(buf, "\r\n", &save);
    while (tok) {
        if (tok[0] != '\0') {
            if (n == cap) {
                cap *= 2;
                char** nl = (char**)realloc(lines, (size_t)cap * sizeof(char*));
                if (!nl) break;
                lines = nl;
            }
            lines[n++] = strdup(tok);
        }
        tok = strtok_r(NULL, "\r\n", &save);
    }

    free(buf);
    *out_lines = lines;
    *out_n = n;
    return n > 0;
}

static bool ensure_input_file(void) {
    if (access("input.txt", R_OK) == 0) return true;
    const char* url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt";
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "curl -fsSL \"%s\" -o input.txt >/dev/null 2>&1", url);
    int rc = system(cmd);
    return rc == 0 && access("input.txt", R_OK) == 0;
}

static void free_lines(char** lines, int n) {
    for (int i = 0; i < n; i++) free(lines[i]);
    free(lines);
}

static void shuffle_lines(char** lines, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char* tmp = lines[i];
        lines[i] = lines[j];
        lines[j] = tmp;
    }
}

static int prepare_example(const char* doc, int BOS, const int* char_to_id, int block_size, int* toks, TrainCache* cache) {
    int len = (int)strlen(doc);
    toks[0] = BOS;
    for (int i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)doc[i];
        toks[i + 1] = char_to_id[ch];
    }
    toks[len + 1] = BOS;

    int n = len + 1;
    if (n > block_size) n = block_size;
    cache->n = n;
    for (int i = 0; i < n; i++) {
        cache->in_tokens[i] = toks[i];
        cache->targets[i] = toks[i + 1];
    }
    return n;
}

static void build_vocab(char** docs, int n_docs, char* id_to_char, int* vocab_size, int* char_to_id) {
    bool seen[256] = {0};
    for (int i = 0; i < n_docs; i++) {
        for (const unsigned char* p = (const unsigned char*)docs[i]; *p; p++) {
            seen[*p] = true;
        }
    }

    int v = 0;
    for (int c = 0; c < 256; c++) {
        if (seen[c]) {
            id_to_char[v] = (char)c;
            char_to_id[c] = v;
            v++;
        }
    }
    *vocab_size = v + 1;  // +1 for BOS
}

static inline float* row(float* m, int cols, int r) {
    return m + (size_t)r * cols;
}

// y = W x, W row-major [out_dim, in_dim]
static void linear(const float* W, const float* x, float* y, int out_dim, int in_dim) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim, 1.0f, W, in_dim, x, 1, 0.0f, y, 1);
}

// y += W^T x, W row-major [out_dim, in_dim], x[out_dim], y[in_dim]
static void linear_t_accum(const float* W, const float* x, float* y, int out_dim, int in_dim) {
    cblas_sgemv(CblasRowMajor, CblasTrans, out_dim, in_dim, 1.0f, W, in_dim, x, 1, 1.0f, y, 1);
}

static void outer_add(float* G, const float* a, const float* b, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float ao = a[o];
        float* go = G + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) go[i] += ao * b[i];
    }
}

static void rmsnorm_forward(const float* x, float* y, int n) {
    float ss = 0.0f;
    vDSP_svesq(x, 1, &ss, n);
    float ms = ss / (float)n;
    float scale = 1.0f / sqrtf(ms + 1e-5f);
    vDSP_vsmul(x, 1, &scale, y, 1, n);
}

static void rmsnorm_backward(const float* x, const float* dy, float* dx, int n) {
    float ss = 0.0f;
    vDSP_svesq(x, 1, &ss, n);
    float ms = ss / (float)n;
    float s = 1.0f / sqrtf(ms + 1e-5f);

    float dot = cblas_sdot(n, dy, 1, x, 1);
    float coeff = -(s * s * s) * dot / (float)n;

    for (int i = 0; i < n; i++) dx[i] += dy[i] * s + x[i] * coeff;
}

static void softmax_forward(const float* logits, float* probs, int n, float temperature) {
    float inv_t = 1.0f / fmaxf(temperature, 1e-4f);
    for (int i = 0; i < n; i++) probs[i] = logits[i] * inv_t;

    float mx = -INFINITY;
    vDSP_maxv(probs, 1, &mx, n);

    for (int i = 0; i < n; i++) probs[i] = expf(probs[i] - mx);

    float s = 0.0f;
    vDSP_sve(probs, 1, &s, n);
    float inv_s = 1.0f / fmaxf(s, 1e-9f);
    vDSP_vsmul(probs, 1, &inv_s, probs, 1, n);
}

static int sample_from_probs(const float* probs, int n) {
    float r = rand_uniform();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (r <= cdf) return i;
    }
    return n - 1;
}

static size_t param_count(const Config* cfg) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;
    return V * E + B * E + V * E + 4 * (L * E * E) + L * (4 * E) * E + L * E * (4 * E);
}

static void alloc_weights(const Config* cfg, Weights* w) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;

    w->wte = (float*)malloc(V * E * sizeof(float));
    w->wpe = (float*)malloc(B * E * sizeof(float));
    w->lm_head = (float*)malloc(V * E * sizeof(float));

    w->attn_wq = (float*)malloc(L * E * E * sizeof(float));
    w->attn_wk = (float*)malloc(L * E * E * sizeof(float));
    w->attn_wv = (float*)malloc(L * E * E * sizeof(float));
    w->attn_wo = (float*)malloc(L * E * E * sizeof(float));

    w->mlp_fc1 = (float*)malloc(L * (4 * E) * E * sizeof(float));
    w->mlp_fc2 = (float*)malloc(L * E * (4 * E) * sizeof(float));
}

static void free_weights(Weights* w) {
    free(w->wte);
    free(w->wpe);
    free(w->lm_head);
    free(w->attn_wq);
    free(w->attn_wk);
    free(w->attn_wv);
    free(w->attn_wo);
    free(w->mlp_fc1);
    free(w->mlp_fc2);
}

static void init_weights(const Config* cfg, Weights* w) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;

    init_mat(w->wte, V * E, 0.02f);
    init_mat(w->wpe, B * E, 0.02f);
    init_mat(w->lm_head, V * E, 0.02f);

    init_mat(w->attn_wq, L * E * E, 0.02f);
    init_mat(w->attn_wk, L * E * E, 0.02f);
    init_mat(w->attn_wv, L * E * E, 0.02f);
    zero_mat(w->attn_wo, L * E * E);

    init_mat(w->mlp_fc1, L * (4 * E) * E, 0.02f);
    zero_mat(w->mlp_fc2, L * E * (4 * E));
}

static void alloc_grads(const Config* cfg, Grads* g) {
    alloc_weights(cfg, (Weights*)g);
}

static void zero_grads(const Config* cfg, Grads* g) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;

    zero_mat(g->wte, V * E);
    zero_mat(g->wpe, B * E);
    zero_mat(g->lm_head, V * E);

    zero_mat(g->attn_wq, L * E * E);
    zero_mat(g->attn_wk, L * E * E);
    zero_mat(g->attn_wv, L * E * E);
    zero_mat(g->attn_wo, L * E * E);

    zero_mat(g->mlp_fc1, L * (4 * E) * E);
    zero_mat(g->mlp_fc2, L * E * (4 * E));
}

static void free_grads(Grads* g) {
    free_weights((Weights*)g);
}

static void alloc_adam(const Config* cfg, AdamBuf* b) {
    alloc_weights(cfg, (Weights*)b);
    zero_grads(cfg, (Grads*)b);
}

static void free_adam(AdamBuf* b) {
    free_weights((Weights*)b);
}

static void alloc_train_cache(const Config* cfg, TrainCache* c) {
    int T = cfg->block_size;
    int E = cfg->n_embd;
    int L = cfg->n_layer;
    int H = cfg->n_head;
    int V = cfg->vocab_size;

    c->n = 0;
    c->in_tokens = (int*)malloc((size_t)T * sizeof(int));
    c->targets = (int*)malloc((size_t)T * sizeof(int));

    c->embed_sum = (float*)malloc((size_t)T * E * sizeof(float));
    c->x0 = (float*)malloc((size_t)T * E * sizeof(float));
    c->x_final = (float*)malloc((size_t)T * E * sizeof(float));
    c->logits = (float*)malloc((size_t)T * V * sizeof(float));

    c->x_in = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->xn_attn = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->q = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->k = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->v = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->attn_out = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->x_after_attn = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->xn_mlp = (float*)malloc((size_t)L * T * E * sizeof(float));
    c->h1 = (float*)malloc((size_t)L * T * (4 * E) * sizeof(float));
    c->h2 = (float*)malloc((size_t)L * T * (4 * E) * sizeof(float));
    c->x_out = (float*)malloc((size_t)L * T * E * sizeof(float));

    c->attn_probs = (float*)calloc((size_t)L * T * H * T, sizeof(float));
}

static void free_train_cache(TrainCache* c) {
    free(c->in_tokens);
    free(c->targets);

    free(c->embed_sum);
    free(c->x0);
    free(c->x_final);
    free(c->logits);

    free(c->x_in);
    free(c->xn_attn);
    free(c->q);
    free(c->k);
    free(c->v);
    free(c->attn_out);
    free(c->x_after_attn);
    free(c->xn_mlp);
    free(c->h1);
    free(c->h2);
    free(c->x_out);
    free(c->attn_probs);
}

static inline float* cache_vec(float* base, int a, int b, int A, int B, int C) {
    (void)A;
    return base + ((size_t)a * B + b) * C;
}

static inline float* cache_head(float* base, int l, int pos, int h, int L, int T, int H, int T2) {
    (void)L;
    return base + (((size_t)l * T + pos) * H + h) * T2;
}

static float forward_sequence(const Config* cfg, const Weights* w, TrainCache* c) {
    int T = c->n;
    int E = cfg->n_embd;
    int H = cfg->n_head;
    int D = cfg->head_dim;
    int L = cfg->n_layer;
    int V = cfg->vocab_size;

    float* scratch_probs = (float*)malloc((size_t)V * sizeof(float));
    float total_loss = 0.0f;

    for (int pos = 0; pos < T; pos++) {
        float* es = row(c->embed_sum, E, pos);
        float* x = row(c->x0, E, pos);

        const float* tok = row(w->wte, E, c->in_tokens[pos]);
        const float* pe = row(w->wpe, E, pos);
        for (int i = 0; i < E; i++) es[i] = tok[i] + pe[i];
        rmsnorm_forward(es, x, E);

        float* x_cur = x;

        for (int li = 0; li < L; li++) {
            float* x_in = cache_vec(c->x_in, li, pos, L, T, E);
            float* xn_attn = cache_vec(c->xn_attn, li, pos, L, T, E);
            float* q = cache_vec(c->q, li, pos, L, T, E);
            float* k = cache_vec(c->k, li, pos, L, T, E);
            float* v = cache_vec(c->v, li, pos, L, T, E);
            float* attn_out = cache_vec(c->attn_out, li, pos, L, T, E);
            float* x_after_attn = cache_vec(c->x_after_attn, li, pos, L, T, E);
            float* xn_mlp = cache_vec(c->xn_mlp, li, pos, L, T, E);
            float* h1 = cache_vec(c->h1, li, pos, L, T, 4 * E);
            float* h2 = cache_vec(c->h2, li, pos, L, T, 4 * E);
            float* x_out = cache_vec(c->x_out, li, pos, L, T, E);

            memcpy(x_in, x_cur, (size_t)E * sizeof(float));
            rmsnorm_forward(x_in, xn_attn, E);

            const float* Wq = w->attn_wq + (size_t)li * E * E;
            const float* Wk = w->attn_wk + (size_t)li * E * E;
            const float* Wv = w->attn_wv + (size_t)li * E * E;
            const float* Wo = w->attn_wo + (size_t)li * E * E;

            linear(Wq, xn_attn, q, E, E);
            linear(Wk, xn_attn, k, E, E);
            linear(Wv, xn_attn, v, E, E);

            for (int i = 0; i < E; i++) attn_out[i] = 0.0f;

            for (int h = 0; h < H; h++) {
                int hs = h * D;
                float* probs = cache_head(c->attn_probs, li, pos, h, L, cfg->block_size, H, cfg->block_size);

                for (int t = 0; t <= pos; t++) {
                    const float* kt = cache_vec(c->k, li, t, L, T, E) + hs;
                    probs[t] = cblas_sdot(D, q + hs, 1, kt, 1) / sqrtf((float)D);
                }
                softmax_forward(probs, probs, pos + 1, 1.0f);

                float* out_h = attn_out + hs;
                for (int j = 0; j < D; j++) out_h[j] = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    const float* vt = cache_vec(c->v, li, t, L, T, E) + hs;
                    cblas_saxpy(D, probs[t], vt, 1, out_h, 1);
                }
            }

            linear(Wo, attn_out, x_after_attn, E, E);
            for (int i = 0; i < E; i++) x_after_attn[i] += x_in[i];

            rmsnorm_forward(x_after_attn, xn_mlp, E);
            const float* W1 = w->mlp_fc1 + (size_t)li * (4 * E) * E;
            const float* W2 = w->mlp_fc2 + (size_t)li * E * (4 * E);

            linear(W1, xn_mlp, h1, 4 * E, E);
            for (int i = 0; i < 4 * E; i++) {
                float r = fmaxf(0.0f, h1[i]);
                h2[i] = r * r;
            }
            linear(W2, h2, x_out, E, 4 * E);
            for (int i = 0; i < E; i++) x_out[i] += x_after_attn[i];

            x_cur = x_out;
        }

        memcpy(row(c->x_final, E, pos), x_cur, (size_t)E * sizeof(float));
        linear(w->lm_head, x_cur, row(c->logits, V, pos), V, E);

        softmax_forward(row(c->logits, V, pos), scratch_probs, V, 1.0f);
        int tgt = c->targets[pos];
        total_loss += -logf(fmaxf(scratch_probs[tgt], 1e-12f));
    }

    free(scratch_probs);
    return total_loss / (float)T;
}

static void backward_sequence(const Config* cfg, const Weights* w, const TrainCache* c, Grads* g) {
    int T = c->n;
    int E = cfg->n_embd;
    int H = cfg->n_head;
    int D = cfg->head_dim;
    int L = cfg->n_layer;
    int V = cfg->vocab_size;

    float invT = 1.0f / (float)T;

    float* d_logits = (float*)malloc((size_t)V * sizeof(float));
    float* d_top = (float*)calloc((size_t)T * E, sizeof(float));

    for (int pos = 0; pos < T; pos++) {
        softmax_forward(row((float*)c->logits, V, pos), d_logits, V, 1.0f);
        d_logits[c->targets[pos]] -= 1.0f;
        for (int i = 0; i < V; i++) d_logits[i] *= invT;

        outer_add(g->lm_head, d_logits, row((float*)c->x_final, E, pos), V, E);
        linear_t_accum(w->lm_head, d_logits, row(d_top, E, pos), V, E);
    }

    float* dK = (float*)calloc((size_t)L * T * E, sizeof(float));
    float* dV = (float*)calloc((size_t)L * T * E, sizeof(float));

    float* d = (float*)malloc((size_t)E * sizeof(float));
    float* tmpE = (float*)malloc((size_t)E * sizeof(float));
    float* d_xin = (float*)malloc((size_t)E * sizeof(float));
    float* d_attn_out = (float*)malloc((size_t)E * sizeof(float));
    float* d_xn = (float*)malloc((size_t)E * sizeof(float));

    float* d_h2 = (float*)malloc((size_t)(4 * E) * sizeof(float));
    float* d_h1 = (float*)malloc((size_t)(4 * E) * sizeof(float));
    float* d_mlp = (float*)malloc((size_t)E * sizeof(float));

    float* d_a = (float*)malloc((size_t)cfg->block_size * sizeof(float));
    float* d_z = (float*)malloc((size_t)cfg->block_size * sizeof(float));

    float* d_embed = (float*)calloc((size_t)E, sizeof(float));
    for (int pos = T - 1; pos >= 0; pos--) {
        memcpy(d, row(d_top, E, pos), (size_t)E * sizeof(float));

        for (int li = L - 1; li >= 0; li--) {
            float* x_in = cache_vec((float*)c->x_in, li, pos, L, T, E);
            float* xn_attn = cache_vec((float*)c->xn_attn, li, pos, L, T, E);
            float* q = cache_vec((float*)c->q, li, pos, L, T, E);
            float* attn_out = cache_vec((float*)c->attn_out, li, pos, L, T, E);
            float* x_after_attn = cache_vec((float*)c->x_after_attn, li, pos, L, T, E);
            float* xn_mlp = cache_vec((float*)c->xn_mlp, li, pos, L, T, E);
            float* h1 = cache_vec((float*)c->h1, li, pos, L, T, 4 * E);
            float* h2 = cache_vec((float*)c->h2, li, pos, L, T, 4 * E);

            const float* Wq = w->attn_wq + (size_t)li * E * E;
            const float* Wk = w->attn_wk + (size_t)li * E * E;
            const float* Wv = w->attn_wv + (size_t)li * E * E;
            const float* Wo = w->attn_wo + (size_t)li * E * E;
            const float* W1 = w->mlp_fc1 + (size_t)li * (4 * E) * E;
            const float* W2 = w->mlp_fc2 + (size_t)li * E * (4 * E);

            // MLP block backward
            memcpy(d_mlp, d, (size_t)E * sizeof(float));
            memcpy(d_xin, d, (size_t)E * sizeof(float));  // residual path to x_after_attn

            outer_add(g->mlp_fc2 + (size_t)li * E * (4 * E), d_mlp, h2, E, 4 * E);
            for (int i = 0; i < 4 * E; i++) d_h2[i] = 0.0f;
            linear_t_accum(W2, d_mlp, d_h2, E, 4 * E);

            for (int i = 0; i < 4 * E; i++) {
                d_h1[i] = (h1[i] > 0.0f) ? (d_h2[i] * 2.0f * h1[i]) : 0.0f;
            }

            outer_add(g->mlp_fc1 + (size_t)li * (4 * E) * E, d_h1, xn_mlp, 4 * E, E);
            for (int i = 0; i < E; i++) d_xn[i] = 0.0f;
            linear_t_accum(W1, d_h1, d_xn, 4 * E, E);

            rmsnorm_backward(x_after_attn, d_xn, d_xin, E);

            // Attention block backward
            memcpy(d_attn_out, d_xin, (size_t)E * sizeof(float));  // through Wo path
            memcpy(tmpE, d_xin, (size_t)E * sizeof(float));         // residual x_in path

            outer_add(g->attn_wo + (size_t)li * E * E, d_attn_out, attn_out, E, E);
            for (int i = 0; i < E; i++) d_attn_out[i] = 0.0f;
            linear_t_accum(Wo, d_xin, d_attn_out, E, E);

            for (int i = 0; i < E; i++) d_xn[i] = 0.0f;

            for (int h = 0; h < H; h++) {
                int hs = h * D;
                const float* qh = q + hs;
                const float* probs = cache_head((float*)c->attn_probs, li, pos, h, L, cfg->block_size, H, cfg->block_size);
                const float* d_head = d_attn_out + hs;

                for (int t = 0; t <= pos; t++) {
                    const float* vt = cache_vec((float*)c->v, li, t, L, T, E) + hs;
                    d_a[t] = cblas_sdot(D, d_head, 1, vt, 1);
                }

                float dot_da_p = 0.0f;
                for (int t = 0; t <= pos; t++) dot_da_p += d_a[t] * probs[t];
                for (int t = 0; t <= pos; t++) d_z[t] = probs[t] * (d_a[t] - dot_da_p);

                float* dq_h = d_xn + hs;
                for (int j = 0; j < D; j++) dq_h[j] = 0.0f;

                for (int t = 0; t <= pos; t++) {
                    const float* kt = cache_vec((float*)c->k, li, t, L, T, E) + hs;
                    float* dkt = cache_vec(dK, li, t, L, T, E) + hs;
                    float* dvt = cache_vec(dV, li, t, L, T, E) + hs;

                    float inv_sqrt_d = 1.0f / sqrtf((float)D);
                    for (int j = 0; j < D; j++) {
                        dq_h[j] += d_z[t] * kt[j] * inv_sqrt_d;
                        dkt[j] += d_z[t] * qh[j] * inv_sqrt_d;
                        dvt[j] += probs[t] * d_head[j];
                    }
                }
            }

            float* dk_cur = cache_vec(dK, li, pos, L, T, E);
            float* dv_cur = cache_vec(dV, li, pos, L, T, E);

            outer_add(g->attn_wq + (size_t)li * E * E, d_xn, xn_attn, E, E);
            linear_t_accum(Wq, d_xn, tmpE, E, E);

            outer_add(g->attn_wk + (size_t)li * E * E, dk_cur, xn_attn, E, E);
            linear_t_accum(Wk, dk_cur, tmpE, E, E);

            outer_add(g->attn_wv + (size_t)li * E * E, dv_cur, xn_attn, E, E);
            linear_t_accum(Wv, dv_cur, tmpE, E, E);

            for (int i = 0; i < E; i++) d[i] = 0.0f;
            rmsnorm_backward(x_in, tmpE, d, E);
        }

        for (int i = 0; i < E; i++) d_embed[i] = 0.0f;
        rmsnorm_backward(row((float*)c->embed_sum, E, pos), d, d_embed, E);

        float* gwte = row(g->wte, E, c->in_tokens[pos]);
        float* gwpe = row(g->wpe, E, pos);
        for (int i = 0; i < E; i++) {
            gwte[i] += d_embed[i];
            gwpe[i] += d_embed[i];
        }
    }
    free(d_embed);

    free(d_logits);
    free(d_top);

    free(dK);
    free(dV);

    free(d);
    free(tmpE);
    free(d_xin);
    free(d_attn_out);
    free(d_xn);
    free(d_h2);
    free(d_h1);
    free(d_mlp);
    free(d_a);
    free(d_z);
}

static void adam_update_array(float* p, const float* g, float* m, float* v, size_t n,
                              float lr_t, float beta1, float beta2, float eps, int step1) {
    float b1t = 1.0f - powf(beta1, (float)step1);
    float b2t = 1.0f - powf(beta2, (float)step1);
    for (size_t i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        float m_hat = m[i] / b1t;
        float v_hat = v[i] / b2t;
        p[i] -= lr_t * m_hat / (sqrtf(v_hat) + eps);
    }
}

static void adam_update(const Config* cfg, Weights* w, const Grads* g, AdamBuf* m1, AdamBuf* m2,
                        float lr_t, float beta1, float beta2, float eps, int step1) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;

    adam_update_array(w->wte, g->wte, m1->wte, m2->wte, V * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->wpe, g->wpe, m1->wpe, m2->wpe, B * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->lm_head, g->lm_head, m1->lm_head, m2->lm_head, V * E, lr_t, beta1, beta2, eps, step1);

    adam_update_array(w->attn_wq, g->attn_wq, m1->attn_wq, m2->attn_wq, L * E * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->attn_wk, g->attn_wk, m1->attn_wk, m2->attn_wk, L * E * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->attn_wv, g->attn_wv, m1->attn_wv, m2->attn_wv, L * E * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->attn_wo, g->attn_wo, m1->attn_wo, m2->attn_wo, L * E * E, lr_t, beta1, beta2, eps, step1);

    adam_update_array(w->mlp_fc1, g->mlp_fc1, m1->mlp_fc1, m2->mlp_fc1, L * (4 * E) * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->mlp_fc2, g->mlp_fc2, m1->mlp_fc2, m2->mlp_fc2, L * E * (4 * E), lr_t, beta1, beta2, eps, step1);
}

static void gpt_infer_step(const Config* cfg, const Weights* w,
                           float* kcache, float* vcache,
                           int token_id, int pos_id, float* logits) {
    int E = cfg->n_embd;
    int H = cfg->n_head;
    int D = cfg->head_dim;
    int L = cfg->n_layer;
    int B = cfg->block_size;

    float* x = (float*)malloc((size_t)E * sizeof(float));
    float* tmp = (float*)malloc((size_t)E * sizeof(float));
    float* xn = (float*)malloc((size_t)E * sizeof(float));
    float* q = (float*)malloc((size_t)E * sizeof(float));
    float* k = (float*)malloc((size_t)E * sizeof(float));
    float* v = (float*)malloc((size_t)E * sizeof(float));
    float* attn_out = (float*)malloc((size_t)E * sizeof(float));
    float* mlp = (float*)malloc((size_t)(4 * E) * sizeof(float));

    const float* tok = row(w->wte, E, token_id);
    const float* pe = row(w->wpe, E, pos_id);
    for (int i = 0; i < E; i++) tmp[i] = tok[i] + pe[i];
    rmsnorm_forward(tmp, x, E);

    for (int li = 0; li < L; li++) {
        memcpy(tmp, x, (size_t)E * sizeof(float));
        rmsnorm_forward(x, xn, E);

        const float* Wq = w->attn_wq + (size_t)li * E * E;
        const float* Wk = w->attn_wk + (size_t)li * E * E;
        const float* Wv = w->attn_wv + (size_t)li * E * E;
        const float* Wo = w->attn_wo + (size_t)li * E * E;

        linear(Wq, xn, q, E, E);
        linear(Wk, xn, k, E, E);
        linear(Wv, xn, v, E, E);

        memcpy(kcache + ((size_t)li * B + pos_id) * E, k, (size_t)E * sizeof(float));
        memcpy(vcache + ((size_t)li * B + pos_id) * E, v, (size_t)E * sizeof(float));

        for (int i = 0; i < E; i++) attn_out[i] = 0.0f;

        float* probs = (float*)malloc((size_t)(pos_id + 1) * sizeof(float));
        for (int h = 0; h < H; h++) {
            int hs = h * D;
            for (int t = 0; t <= pos_id; t++) {
                const float* kt = kcache + ((size_t)li * B + t) * E + hs;
                probs[t] = cblas_sdot(D, q + hs, 1, kt, 1) / sqrtf((float)D);
            }
            softmax_forward(probs, probs, pos_id + 1, 1.0f);
            float* out_h = attn_out + hs;
            for (int j = 0; j < D; j++) out_h[j] = 0.0f;
            for (int t = 0; t <= pos_id; t++) {
                const float* vt = vcache + ((size_t)li * B + t) * E + hs;
                cblas_saxpy(D, probs[t], vt, 1, out_h, 1);
            }
        }
        free(probs);

        linear(Wo, attn_out, x, E, E);
        for (int i = 0; i < E; i++) x[i] += tmp[i];

        memcpy(tmp, x, (size_t)E * sizeof(float));
        rmsnorm_forward(x, xn, E);

        const float* W1 = w->mlp_fc1 + (size_t)li * (4 * E) * E;
        const float* W2 = w->mlp_fc2 + (size_t)li * E * (4 * E);
        linear(W1, xn, mlp, 4 * E, E);
        for (int i = 0; i < 4 * E; i++) {
            float r = fmaxf(0.0f, mlp[i]);
            mlp[i] = r * r;
        }
        linear(W2, mlp, x, E, 4 * E);
        for (int i = 0; i < E; i++) x[i] += tmp[i];
    }

    linear(w->lm_head, x, logits, cfg->vocab_size, E);

    free(x);
    free(tmp);
    free(xn);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(mlp);
}

static float eval_split_loss(const Config* cfg, const Weights* w, TrainCache* cache,
                             char** docs, int start, int count, int BOS,
                             const int* char_to_id, int* toks, int eval_iters) {
    if (count <= 0) return NAN;
    float s = 0.0f;
    for (int i = 0; i < eval_iters; i++) {
        int idx = start + (rand() % count);
        prepare_example(docs[idx], BOS, char_to_id, cfg->block_size, toks, cache);
        s += forward_sequence(cfg, w, cache);
    }
    return s / (float)eval_iters;
}

static bool save_checkpoint(const char* path, const Config* cfg, const Weights* w) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    const uint32_t magic = 0x4D475043;  // MGPC
    const uint32_t version = 1;
    if (fwrite(&magic, sizeof(magic), 1, f) != 1) goto fail;
    if (fwrite(&version, sizeof(version), 1, f) != 1) goto fail;
    if (fwrite(cfg, sizeof(*cfg), 1, f) != 1) goto fail;

    size_t V = (size_t)cfg->vocab_size, E = (size_t)cfg->n_embd, L = (size_t)cfg->n_layer, B = (size_t)cfg->block_size;
    if (fwrite(w->wte, sizeof(float), V * E, f) != V * E) goto fail;
    if (fwrite(w->wpe, sizeof(float), B * E, f) != B * E) goto fail;
    if (fwrite(w->lm_head, sizeof(float), V * E, f) != V * E) goto fail;
    if (fwrite(w->attn_wq, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->attn_wk, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->attn_wv, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->attn_wo, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->mlp_fc1, sizeof(float), L * (4 * E) * E, f) != L * (4 * E) * E) goto fail;
    if (fwrite(w->mlp_fc2, sizeof(float), L * E * (4 * E), f) != L * E * (4 * E)) goto fail;
    fclose(f);
    return true;
fail:
    fclose(f);
    return false;
}

static bool load_checkpoint(const char* path, const Config* cfg, Weights* w) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0;
    Config ck = {0};
    if (fread(&magic, sizeof(magic), 1, f) != 1) goto fail;
    if (fread(&version, sizeof(version), 1, f) != 1) goto fail;
    if (fread(&ck, sizeof(ck), 1, f) != 1) goto fail;
    if (magic != 0x4D475043 || version != 1) goto fail;
    if (memcmp(&ck, cfg, sizeof(Config)) != 0) goto fail;

    size_t V = (size_t)cfg->vocab_size, E = (size_t)cfg->n_embd, L = (size_t)cfg->n_layer, B = (size_t)cfg->block_size;
    if (fread(w->wte, sizeof(float), V * E, f) != V * E) goto fail;
    if (fread(w->wpe, sizeof(float), B * E, f) != B * E) goto fail;
    if (fread(w->lm_head, sizeof(float), V * E, f) != V * E) goto fail;
    if (fread(w->attn_wq, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->attn_wk, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->attn_wv, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->attn_wo, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->mlp_fc1, sizeof(float), L * (4 * E) * E, f) != L * (4 * E) * E) goto fail;
    if (fread(w->mlp_fc2, sizeof(float), L * E * (4 * E), f) != L * E * (4 * E)) goto fail;
    fclose(f);
    return true;
fail:
    fclose(f);
    return false;
}

int main(int argc, char** argv) {
    srand(42);  // align with Python gist

    char** docs = NULL;
    int n_docs = 0;
    if (!ensure_input_file()) {
        fprintf(stderr, "warning: input.txt missing and auto-download failed, using tiny fallback corpus\n");
    }
    if (!load_lines("input.txt", &docs, &n_docs)) {
        static const char* fallback[] = {"anna", "bob", "carol", "david", "emma", "frank"};
        n_docs = (int)(sizeof(fallback) / sizeof(fallback[0]));
        docs = (char**)malloc((size_t)n_docs * sizeof(char*));
        for (int i = 0; i < n_docs; i++) docs[i] = strdup(fallback[i]);
    }
    shuffle_lines(docs, n_docs);

    char id_to_char[256] = {0};
    int char_to_id[256];
    for (int i = 0; i < 256; i++) char_to_id[i] = -1;

    int vocab_size = 0;
    build_vocab(docs, n_docs, id_to_char, &vocab_size, char_to_id);
    int BOS = vocab_size - 1;

    Config cfg = {
        .n_embd = 16,
        .n_head = 4,
        .n_layer = 1,
        .block_size = 8,
        .head_dim = 16 / 4,
        .vocab_size = vocab_size,
    };

    int num_steps = 500;
    float temperature = 0.5f;
    int num_samples = 20;
    int eval_interval = 100;
    int eval_iters = 100;
    if (argc > 1) num_steps = atoi(argv[1]);
    if (argc > 2) temperature = atof(argv[2]);
    if (argc > 3) num_samples = atoi(argv[3]);
    if (argc > 4) eval_interval = atoi(argv[4]);
    if (argc > 5) eval_iters = atoi(argv[5]);

    Weights w = {0};
    alloc_weights(&cfg, &w);
    init_weights(&cfg, &w);

    Grads g = {0};
    alloc_grads(&cfg, &g);

    AdamBuf m1 = {0}, m2 = {0};
    alloc_adam(&cfg, &m1);
    alloc_adam(&cfg, &m2);

    TrainCache cache = {0};
    alloc_train_cache(&cfg, &cache);

    printf("num docs: %d\n", n_docs);
    printf("vocab size: %d\n", cfg.vocab_size);
    printf("num params: %zu\n", param_count(&cfg));

    float learning_rate = 1e-2f, beta1 = 0.9f, beta2 = 0.95f, eps_adam = 1e-8f;
    int n_val = n_docs / 10;
    if (n_docs > 1 && n_val < 1) n_val = 1;
    if (n_docs > 1 && n_val >= n_docs) n_val = n_docs - 1;
    if (n_docs <= 1) n_val = 0;
    int n_train = n_docs - n_val;
    printf("train docs: %d | val docs: %d\n", n_train, n_val);
    int* toks = (int*)malloc((size_t)(cfg.block_size + 2) * sizeof(int));

    float* losses = (float*)malloc((size_t)num_steps * sizeof(float));
    float best_val = INFINITY;
    const char* best_ckpt = "ckpt_best.bin";
    FILE* ef = fopen("eval.csv", "w");
    if (ef) fprintf(ef, "step,train_loss,val_loss,lr\n");
    struct timespec t0 = {0}, t1 = {0};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int step = 0; step < num_steps; step++) {
        int idx = rand() % n_train;
        prepare_example(docs[idx], BOS, char_to_id, cfg.block_size, toks, &cache);

        zero_grads(&cfg, &g);
        float loss = forward_sequence(&cfg, &w, &cache);
        backward_sequence(&cfg, &w, &cache, &g);

        float lr_t = learning_rate * 0.5f * (1.0f + cosf((float)M_PI * (float)step / (float)num_steps));
        adam_update(&cfg, &w, &g, &m1, &m2, lr_t, beta1, beta2, eps_adam, step + 1);

        losses[step] = loss;
        printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss);

        bool do_eval = ((step + 1) % eval_interval == 0) || (step + 1 == num_steps);
        if (do_eval) {
            float train_eval = eval_split_loss(&cfg, &w, &cache, docs, 0, n_train, BOS, char_to_id, toks, eval_iters);
            float val_eval = (n_val > 0) ? eval_split_loss(&cfg, &w, &cache, docs, n_train, n_val, BOS, char_to_id, toks, eval_iters) : NAN;
            printf("eval step %4d | train %.4f | val %.4f | lr %.6f\n", step + 1, train_eval, val_eval, lr_t);
            if (ef) fprintf(ef, "%d,%.8f,%.8f,%.8f\n", step + 1, train_eval, val_eval, lr_t);

            float metric = (n_val > 0) ? val_eval : train_eval;
            if (metric < best_val) {
                best_val = metric;
                if (save_checkpoint(best_ckpt, &cfg, &w)) {
                    printf("saved new best checkpoint: %s (metric=%.4f)\n", best_ckpt, metric);
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (ef) fclose(ef);

    FILE* lf = fopen("losses.csv", "w");
    if (lf) {
        fprintf(lf, "step,loss\n");
        for (int i = 0; i < num_steps; i++) fprintf(lf, "%d,%.8f\n", i + 1, losses[i]);
        fclose(lf);
    }
    free(losses);
    double train_sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    double steps_per_sec = (train_sec > 0.0) ? ((double)num_steps / train_sec) : 0.0;
    printf("train_time_sec: %.6f | train_steps_per_sec: %.2f\n", train_sec, steps_per_sec);

    if (load_checkpoint(best_ckpt, &cfg, &w)) {
        printf("loaded best checkpoint: %s\n", best_ckpt);
    }

    printf("\n--- inference ---\n");
    float* logits = (float*)malloc((size_t)cfg.vocab_size * sizeof(float));
    float* probs = (float*)malloc((size_t)cfg.vocab_size * sizeof(float));
    float* kcache = (float*)calloc((size_t)cfg.n_layer * cfg.block_size * cfg.n_embd, sizeof(float));
    float* vcache = (float*)calloc((size_t)cfg.n_layer * cfg.block_size * cfg.n_embd, sizeof(float));

    for (int s = 0; s < num_samples; s++) {
        memset(kcache, 0, (size_t)cfg.n_layer * cfg.block_size * cfg.n_embd * sizeof(float));
        memset(vcache, 0, (size_t)cfg.n_layer * cfg.block_size * cfg.n_embd * sizeof(float));

        int token = BOS;
        printf("sample %2d: ", s + 1);
        for (int pos = 0; pos < cfg.block_size; pos++) {
            gpt_infer_step(&cfg, &w, kcache, vcache, token, pos, logits);
            softmax_forward(logits, probs, cfg.vocab_size, temperature);
            token = sample_from_probs(probs, cfg.vocab_size);
            if (token == BOS) break;
            putchar(id_to_char[token]);
        }
        putchar('\n');
    }

    free(logits);
    free(probs);
    free(kcache);
    free(vcache);
    free(toks);

    free_train_cache(&cache);
    free_adam(&m1);
    free_adam(&m2);
    free_grads(&g);
    free_weights(&w);
    free_lines(docs, n_docs);
    return 0;
}
