#include <Accelerate/Accelerate.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int n_embd;
    int n_head;
    int n_layer;
    int block_size;
    int head_dim;
    int vocab_size;
    int ffn_dim;
    int norm_kind;
    int tie_embeddings;
    int act_kind;
    float dropout_p;
} Config;

typedef struct {
    int n_embd;
    int n_head;
    int n_layer;
    int block_size;
    int head_dim;
    int vocab_size;
} ConfigV1;

typedef struct {
    int n_embd;
    int n_head;
    int n_layer;
    int block_size;
    int head_dim;
    int vocab_size;
    int norm_kind;
    int tie_embeddings;
} ConfigV2;

typedef struct {
    int n_embd;
    int n_head;
    int n_layer;
    int block_size;
    int head_dim;
    int vocab_size;
    int norm_kind;
    int tie_embeddings;
    int act_kind;
    float dropout_p;
} ConfigV3;

enum {
    NORM_RMS = 0,
    NORM_LAYER = 1,
};

enum {
    ACT_RELU2 = 0,
    ACT_GELU = 1,
    ACT_RELU = 2,
};

typedef struct {
    float* wte;      // [V, E]
    float* wpe;      // [B, E]
    float* lm_head;  // [V, E]

    float* attn_wq;  // [L, E, E]
    float* attn_wk;  // [L, E, E]
    float* attn_wv;  // [L, E, E]
    float* attn_wo;  // [L, E, E]

    float* mlp_fc1;  // [L, M, E]
    float* mlp_fc2;  // [L, E, M]
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
    float* dropout_mask;  // [L, n, 4E]
    float* x_out;         // [L, n, E]

    float* attn_probs;    // [L, n, H, n]
} TrainCache;

typedef struct {
    uint16_t* train_tokens;
    uint16_t* val_tokens;
    size_t n_train_tokens;
    size_t n_val_tokens;
    int vocab_size;
    int pad_id;
    int user_id;
    int assistant_id;
    int end_id;
} TokenCorpus;

static const char* kChatUserTag = "<|user|>";
static const char* kChatAssistantTag = "<|assistant|>";
static const char* kChatEndTag = "<|end|>";
static const char* kImStartTag = "<|im_start|>";

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

static bool dataset_looks_like_chat(char** docs, int n_docs) {
    int limit = n_docs < 8 ? n_docs : 8;
    for (int i = 0; i < limit; i++) {
        if ((strstr(docs[i], kChatUserTag) && strstr(docs[i], kChatAssistantTag)) ||
            (strstr(docs[i], kImStartTag) && strstr(docs[i], "assistant"))) {
            return true;
        }
    }
    return false;
}

static bool path_has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    if (m > n) return false;
    return strcmp(s + (n - m), suffix) == 0;
}

static const char* norm_kind_name(int norm_kind) {
    return norm_kind == NORM_LAYER ? "layernorm" : "rmsnorm";
}

static const char* act_kind_name(int act_kind) {
    if (act_kind == ACT_GELU) return "gelu";
    if (act_kind == ACT_RELU) return "relu";
    return "relu2";
}

static int mlp_dim(const Config* cfg) {
    return cfg->ffn_dim > 0 ? cfg->ffn_dim : (4 * cfg->n_embd);
}

static bool read_text_file(const char* path, char** out_text) {
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
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        return false;
    }
    fclose(f);
    buf[sz] = '\0';
    *out_text = buf;
    return true;
}

static int json_int_field(const char* text, const char* key, int fallback) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(text, needle);
    if (!p) return fallback;
    p = strchr(p, ':');
    if (!p) return fallback;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}

static bool json_string_field(const char* text, const char* key, char* out, size_t out_sz) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(text, needle);
    if (!p) return false;
    p = strchr(p, ':');
    if (!p) return false;
    p = strchr(p, '"');
    if (!p) return false;
    p++;
    const char* q = strchr(p, '"');
    if (!q) return false;
    size_t n = (size_t)(q - p);
    if (n + 1 > out_sz) n = out_sz - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    return true;
}

static void infer_token_paths(const char* train_path, char* eval_path, size_t eval_sz,
                              char* meta_path, size_t meta_sz) {
    snprintf(eval_path, eval_sz, "%s", train_path);
    snprintf(meta_path, meta_sz, "%s", train_path);

    char* eval_name = strrchr(eval_path, '/');
    eval_name = eval_name ? eval_name + 1 : eval_path;
    snprintf(eval_name, eval_sz - (size_t)(eval_name - eval_path), "eval.bin");

    char* meta_name = strrchr(meta_path, '/');
    meta_name = meta_name ? meta_name + 1 : meta_path;
    snprintf(meta_name, meta_sz - (size_t)(meta_name - meta_path), "meta.json");
}

static void infer_parent_dir(const char* path, char* out, size_t out_sz) {
    const char* slash = strrchr(path, '/');
    if (!slash) {
        snprintf(out, out_sz, ".");
        return;
    }
    size_t n = (size_t)(slash - path);
    if (n >= out_sz) n = out_sz - 1;
    memcpy(out, path, n);
    out[n] = '\0';
}

static bool load_token_file_u16(const char* path, uint16_t** out_tokens, size_t* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0 || (sz % (long)sizeof(uint16_t)) != 0) {
        fclose(f);
        return false;
    }
    size_t n = (size_t)sz / sizeof(uint16_t);
    uint16_t* buf = (uint16_t*)malloc(n * sizeof(uint16_t));
    if (!buf) {
        fclose(f);
        return false;
    }
    if (fread(buf, sizeof(uint16_t), n, f) != n) {
        fclose(f);
        free(buf);
        return false;
    }
    fclose(f);
    *out_tokens = buf;
    *out_n = n;
    return true;
}

static bool load_token_corpus(const char* train_path, TokenCorpus* corpus) {
    char eval_path[1024];
    char meta_path[1024];
    char dtype[32];
    infer_token_paths(train_path, eval_path, sizeof(eval_path), meta_path, sizeof(meta_path));

    char* meta_text = NULL;
    if (!read_text_file(meta_path, &meta_text)) return false;
    if (!json_string_field(meta_text, "dtype", dtype, sizeof(dtype))) {
        free(meta_text);
        return false;
    }
    if (strcmp(dtype, "u16") != 0) {
        fprintf(stderr, "error: only u16 token datasets are currently supported\n");
        free(meta_text);
        return false;
    }

    corpus->vocab_size = json_int_field(meta_text, "vocab_size", -1);
    corpus->pad_id = json_int_field(meta_text, "pad_id", 0);
    corpus->user_id = json_int_field(meta_text, "user_id", json_int_field(meta_text, "im_start_id", -1));
    corpus->assistant_id = json_int_field(meta_text, "assistant_id", -1);
    corpus->end_id = json_int_field(meta_text, "end_id", json_int_field(meta_text, "im_end_id", -1));
    free(meta_text);

    if (corpus->vocab_size <= 0) return false;
    if (!load_token_file_u16(train_path, &corpus->train_tokens, &corpus->n_train_tokens)) return false;
    if (!load_token_file_u16(eval_path, &corpus->val_tokens, &corpus->n_val_tokens)) {
        free(corpus->train_tokens);
        corpus->train_tokens = NULL;
        return false;
    }
    return corpus->n_train_tokens > 1 && corpus->n_val_tokens > 1;
}

static void free_token_corpus(TokenCorpus* corpus) {
    free(corpus->train_tokens);
    free(corpus->val_tokens);
    memset(corpus, 0, sizeof(*corpus));
}

static const char* pick_str(const char* const* items, int count) {
    return items[rand() % count];
}

static void sanitize_chat_text(const char* src, char* dst, size_t dst_sz) {
    size_t j = 0;
    for (size_t i = 0; src[i] != '\0' && j + 1 < dst_sz; i++) {
        unsigned char ch = (unsigned char)src[i];
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        dst[j++] = (char)tolower(ch);
    }
    dst[j] = '\0';
}

static void format_chat_prompt(const char* user_input, char* out, size_t out_sz) {
    char clean[512];
    sanitize_chat_text(user_input, clean, sizeof(clean));
    snprintf(out, out_sz, "%s %s %s", kChatUserTag, clean, kChatAssistantTag);
}

static void truncate_to_tail(const char* src, int max_chars, char* out, size_t out_sz) {
    size_t len = strlen(src);
    const char* tail = src;
    if ((int)len > max_chars) tail = src + (len - (size_t)max_chars);
    snprintf(out, out_sz, "%s", tail);
}

static void build_guppy_sample(char* out, size_t out_sz) {
    static const char* const greetings[] = {
        "hi guppy", "hello little fish", "good morning guppy", "hey fish", "hi tiny swimmer"
    };
    static const char* const greeting_replies[] = {
        "hello. the water feels nice today.",
        "hi. i was looking at the light.",
        "hello friend. i am doing slow circles.",
        "hi there. i found a good bubble spot.",
        "hello. my fins are ready for the day."
    };
    static const char* const food_prompts[] = {
        "are you hungry", "do you want food", "what do you eat", "is it snack time", "are you thinking about food"
    };
    static const char* const foods[] = {
        "flakes", "tiny pellets", "brine shrimp", "crumbly food", "little floating snacks"
    };
    static const char* const bubble_prompts[] = {
        "do you like bubbles", "what do bubbles feel like", "are the bubbles loud", "why are you following bubbles"
    };
    static const char* const bubble_replies[] = {
        "i love bubbles. they tickle the water.",
        "bubbles feel funny on my face.",
        "i follow them because they go up and i do not.",
        "the bubbles sound busy but friendly."
    };
    static const char* const light_prompts[] = {
        "is the light too bright", "do you like the light", "what do you think about the lamp", "why are you near the light"
    };
    static const char* const light_things[] = {
        "lamp", "bright top light", "morning light", "soft light", "glass shine"
    };
    static const char* const sleep_prompts[] = {
        "are you sleepy", "goodnight guppy", "do fish sleep", "why are you so still"
    };
    static const char* const sleep_places[] = {
        "plant", "corner", "rock", "filter shadow", "warm side of the tank"
    };
    static const char* const cat_prompts[] = {
        "the cat is looking at you", "are you scared of the cat", "what do you think about the furry thing", "did the cat come back"
    };
    static const char* const rain_prompts[] = {
        "it is raining outside", "do you know what rain is", "can you hear the rain", "is rain scary"
    };
    static const char* const joke_prompts[] = {
        "tell me a joke", "say something funny", "do fish know jokes", "make me laugh"
    };
    static const char* const joke_lines[] = {
        "what did the fish say when it hit the wall. dam.",
        "i told the snail a joke. it laughed very slowly.",
        "my best joke is still about food.",
        "i am funny when i miss the pellet and pretend i did not."
    };
    static const char* const love_prompts[] = {
        "do you love me", "am i your friend", "do you like me", "are we friends"
    };
    static const char* const meaning_prompts[] = {
        "what is the meaning of life", "why are we here", "what matters most", "what is important to you"
    };
    static const char* const reflection_prompts[] = {
        "is that you in the glass", "why do you look at your reflection", "do you know the fish in the glass"
    };
    static const char* const music_prompts[] = {
        "do you like music", "can fish hear songs", "what do you think about this music"
    };
    static const char* const lonely_prompts[] = {
        "are you lonely", "do you need a friend", "do you get bored"
    };
    static const char* const weather_prompts[] = {
        "is it warm today", "how does the water feel", "do you like cold water", "what is the tank like today"
    };
    static const char* const clean_prompts[] = {
        "why is the water changing", "are you okay when i clean the tank", "do you like tank cleaning", "what happens on cleaning day"
    };
    static const char* const clean_replies[] = {
        "cleaning day feels strange at first, then the water tastes bright again.",
        "i act annoyed, but i do like when the glass gets clear.",
        "the net is rude, but fresh water is worth the drama."
    };
    static const char* const snail_prompts[] = {
        "what do you think about the snail", "are the snails your friends", "do you talk to the snail", "why do you watch the snail"
    };
    static const char* const snail_replies[] = {
        "the snail moves like a slow thought. i respect that.",
        "we are not close friends, but i admire the patience.",
        "the snail never answers, so i do most of the talking."
    };
    static const char* const plant_prompts[] = {
        "which plant do you like", "why are you hiding in the plant", "do you sleep by the leaves", "what do the plants feel like"
    };
    static const char* const plant_replies[] = {
        "the soft plant is best. it moves like a quiet curtain.",
        "the leaves make a good hiding place when the room gets loud.",
        "plants make the tank feel less like a bowl and more like a small world."
    };
    static const char* const dream_prompts[] = {
        "do fish dream", "what do you dream about", "did you have a dream", "what happens when you sleep"
    };
    static const char* const dream_replies[] = {
        "if fish dream, mine are mostly bubbles and impossible snacks.",
        "i think i dream about swimming through rooms with no glass.",
        "sometimes i wake up sure i was chasing a shining pellet."
    };
    static const char* const advice_prompts[] = {
        "give me advice", "what should i do today", "say something wise", "how do i stay calm"
    };
    static const char* const advice_replies[] = {
        "move a little slower. most things look less scary that way.",
        "eat when food comes, rest when the water is calm, ignore the useless panic.",
        "find one good corner and remember it is still there when the day feels noisy."
    };
    static const char* const outside_prompts[] = {
        "what do you think about outside", "do you want to leave the tank", "is the room strange", "what is beyond the glass"
    };
    static const char* const outside_replies[] = {
        "outside looks huge and dry. i prefer to study it from here.",
        "the room is interesting, but i am built for the water part of reality.",
        "beyond the glass is mystery. inside the glass is lunch, so i stay practical."
    };
    static const char* const memory_prompts[] = {
        "what do you remember", "do you remember me", "what happened yesterday", "what sticks in your mind"
    };
    static const char* const memory_replies[] = {
        "i remember feeding time very clearly and almost nothing in between.",
        "i remember your face near the glass and the sound before food drops.",
        "my memory is mostly little flashes: light, pellets, bubbles, sleep."
    };
    static const char* const compliment_prompts[] = {
        "you look cute", "your fins are pretty", "you are a lovely fish", "you are beautiful"
    };
    static const char* const compliment_replies[] = {
        "thank you. i grew these fins myself.",
        "that is kind. i was hoping the light would catch the color today.",
        "i accept this praise with grace and a small proud turn."
    };
    static const char* const fear_prompts[] = {
        "what scares you", "are you afraid of anything", "do loud sounds scare you", "what makes you hide"
    };
    static const char* const fear_replies[] = {
        "fast shadows, loud taps, and surprise nets. those are my top concerns.",
        "i do not like sudden noise. it makes the whole tank feel jumpy.",
        "fear feels like cold water inside warm water. i hide until it passes."
    };

    const char* user = "hi guppy";
    char assistant[512];
    int topic = rand() % 22;
    switch (topic) {
        case 0:
            user = pick_str(greetings, (int)(sizeof(greetings) / sizeof(greetings[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(greeting_replies, (int)(sizeof(greeting_replies) / sizeof(greeting_replies[0]))));
            break;
        case 1:
            user = pick_str(food_prompts, (int)(sizeof(food_prompts) / sizeof(food_prompts[0])));
            snprintf(assistant, sizeof(assistant), "yes. always yes. i am thinking about %s right now.",
                     pick_str(foods, (int)(sizeof(foods) / sizeof(foods[0]))));
            break;
        case 2:
            user = pick_str(bubble_prompts, (int)(sizeof(bubble_prompts) / sizeof(bubble_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(bubble_replies, (int)(sizeof(bubble_replies) / sizeof(bubble_replies[0]))));
            break;
        case 3:
            user = pick_str(light_prompts, (int)(sizeof(light_prompts) / sizeof(light_prompts[0])));
            snprintf(assistant, sizeof(assistant), "the %s is nice when it is not too sharp. i like to float under it.",
                     pick_str(light_things, (int)(sizeof(light_things) / sizeof(light_things[0]))));
            break;
        case 4:
            user = pick_str(sleep_prompts, (int)(sizeof(sleep_prompts) / sizeof(sleep_prompts[0])));
            snprintf(assistant, sizeof(assistant), "when i get sleepy i hide by the %s and move only a little.",
                     pick_str(sleep_places, (int)(sizeof(sleep_places) / sizeof(sleep_places[0]))));
            break;
        case 5:
            user = pick_str(cat_prompts, (int)(sizeof(cat_prompts) / sizeof(cat_prompts[0])));
            snprintf(assistant, sizeof(assistant), "i do not trust the furry one. i stay close to the plant when it stares.");
            break;
        case 6:
            user = pick_str(rain_prompts, (int)(sizeof(rain_prompts) / sizeof(rain_prompts[0])));
            snprintf(assistant, sizeof(assistant), "rain sounds soft from inside the tank. it makes outside feel far away.");
            break;
        case 7:
            user = pick_str(joke_prompts, (int)(sizeof(joke_prompts) / sizeof(joke_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(joke_lines, (int)(sizeof(joke_lines) / sizeof(joke_lines[0]))));
            break;
        case 8:
            user = pick_str(love_prompts, (int)(sizeof(love_prompts) / sizeof(love_prompts[0])));
            snprintf(assistant, sizeof(assistant), "yes. you are my favorite big shape outside the glass.");
            break;
        case 9:
            user = pick_str(meaning_prompts, (int)(sizeof(meaning_prompts) / sizeof(meaning_prompts[0])));
            snprintf(assistant, sizeof(assistant), "food first. bubbles second. after that i will think more.");
            break;
        case 10:
            user = pick_str(reflection_prompts, (int)(sizeof(reflection_prompts) / sizeof(reflection_prompts[0])));
            snprintf(assistant, sizeof(assistant), "the glass fish copies me. i still have questions about that.");
            break;
        case 11:
            if (rand() & 1) {
                user = pick_str(music_prompts, (int)(sizeof(music_prompts) / sizeof(music_prompts[0])));
                snprintf(assistant, sizeof(assistant), "some music feels like tiny waves. i like the slow kind.");
            } else if (rand() & 1) {
                user = pick_str(lonely_prompts, (int)(sizeof(lonely_prompts) / sizeof(lonely_prompts[0])));
                snprintf(assistant, sizeof(assistant), "sometimes i watch the snails when the tank feels quiet.");
            } else {
                user = pick_str(weather_prompts, (int)(sizeof(weather_prompts) / sizeof(weather_prompts[0])));
                snprintf(assistant, sizeof(assistant), "the water feels calm and a little warm. that is a good tank day.");
            }
            break;
        case 12:
            user = pick_str(clean_prompts, (int)(sizeof(clean_prompts) / sizeof(clean_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(clean_replies, (int)(sizeof(clean_replies) / sizeof(clean_replies[0]))));
            break;
        case 13:
            user = pick_str(snail_prompts, (int)(sizeof(snail_prompts) / sizeof(snail_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(snail_replies, (int)(sizeof(snail_replies) / sizeof(snail_replies[0]))));
            break;
        case 14:
            user = pick_str(plant_prompts, (int)(sizeof(plant_prompts) / sizeof(plant_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(plant_replies, (int)(sizeof(plant_replies) / sizeof(plant_replies[0]))));
            break;
        case 15:
            user = pick_str(dream_prompts, (int)(sizeof(dream_prompts) / sizeof(dream_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(dream_replies, (int)(sizeof(dream_replies) / sizeof(dream_replies[0]))));
            break;
        case 16:
            user = pick_str(advice_prompts, (int)(sizeof(advice_prompts) / sizeof(advice_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(advice_replies, (int)(sizeof(advice_replies) / sizeof(advice_replies[0]))));
            break;
        case 17:
            user = pick_str(outside_prompts, (int)(sizeof(outside_prompts) / sizeof(outside_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(outside_replies, (int)(sizeof(outside_replies) / sizeof(outside_replies[0]))));
            break;
        case 18:
            user = pick_str(memory_prompts, (int)(sizeof(memory_prompts) / sizeof(memory_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(memory_replies, (int)(sizeof(memory_replies) / sizeof(memory_replies[0]))));
            break;
        case 19:
            user = pick_str(compliment_prompts, (int)(sizeof(compliment_prompts) / sizeof(compliment_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(compliment_replies, (int)(sizeof(compliment_replies) / sizeof(compliment_replies[0]))));
            break;
        case 20:
            user = pick_str(fear_prompts, (int)(sizeof(fear_prompts) / sizeof(fear_prompts[0])));
            snprintf(assistant, sizeof(assistant), "%s",
                     pick_str(fear_replies, (int)(sizeof(fear_replies) / sizeof(fear_replies[0]))));
            break;
        default:
            user = pick_str(weather_prompts, (int)(sizeof(weather_prompts) / sizeof(weather_prompts[0])));
            snprintf(assistant, sizeof(assistant),
                     "today feels gentle. i did a slow lap, watched the room, and decided this is a good water day.");
            break;
    }

    snprintf(out, out_sz, "%s %s %s %s %s", kChatUserTag, user, kChatAssistantTag, assistant, kChatEndTag);
}

static bool write_guppy_dataset(const char* path, int n_samples) {
    FILE* f = fopen(path, "w");
    if (!f) return false;
    char line[1024];
    for (int i = 0; i < n_samples; i++) {
        build_guppy_sample(line, sizeof(line));
        fprintf(f, "%s\n", line);
    }
    fclose(f);
    return true;
}

static int prepare_example(const char* doc, int BOS, const int* char_to_id, int block_size, int* toks, TrainCache* cache) {
    int len = (int)strlen(doc);
    int n = len + 1;
    if (n > block_size) n = block_size;

    toks[0] = BOS;
    int doc_chars = (n < len) ? n : len;
    for (int i = 0; i < doc_chars; i++) {
        unsigned char ch = (unsigned char)doc[i];
        toks[i + 1] = char_to_id[ch];
    }
    if (n == len + 1) toks[n] = BOS;

    cache->n = n;
    for (int i = 0; i < n; i++) {
        cache->in_tokens[i] = toks[i];
        cache->targets[i] = toks[i + 1];
    }
    return n;
}

static int prepare_token_example(const uint16_t* tokens, size_t n_tokens, int block_size, TrainCache* cache) {
    if (n_tokens < 2) return 0;
    int n = block_size;
    if ((size_t)(n + 1) > n_tokens) n = (int)n_tokens - 1;
    size_t max_start = n_tokens - (size_t)(n + 1);
    size_t start = (max_start > 0) ? (size_t)(rand() % (int)(max_start + 1)) : 0;
    cache->n = n;
    for (int i = 0; i < n; i++) {
        cache->in_tokens[i] = tokens[start + (size_t)i];
        cache->targets[i] = tokens[start + (size_t)i + 1];
    }
    return n;
}

static int parse_token_id_list(const char* text, int vocab_size, int* ids, int max_ids) {
    char* copy = strdup(text);
    if (!copy) return -1;

    int n = 0;
    char* save = NULL;
    char* tok = strtok_r(copy, ", \t\r\n", &save);
    while (tok) {
        if (n >= max_ids) {
            free(copy);
            return -1;
        }
        char* end = NULL;
        long v = strtol(tok, &end, 10);
        if (tok[0] == '\0' || !end || end[0] != '\0' || v < 0 || v >= vocab_size) {
            free(copy);
            return -1;
        }
        ids[n++] = (int)v;
        tok = strtok_r(NULL, ", \t\r\n", &save);
    }

    free(copy);
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

// Y[m, out_dim] = X[m, in_dim] * W[out_dim, in_dim]^T
static void linear_seq(const float* W, const float* X, float* Y, int m, int out_dim, int in_dim) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, out_dim, in_dim,
                1.0f, X, in_dim, W, in_dim, 0.0f, Y, out_dim);
}

// Y[m, in_dim] = X[m, out_dim] * W[out_dim, in_dim]
static void linear_t_seq(const float* W, const float* X, float* Y, int m, int out_dim, int in_dim) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, in_dim, out_dim,
                1.0f, X, out_dim, W, in_dim, 0.0f, Y, in_dim);
}

// G[out_dim, in_dim] += dY[m, out_dim]^T * X[m, in_dim]
static void outer_add_seq(float* G, const float* dY, const float* X, int m, int out_dim, int in_dim) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                out_dim, in_dim, m,
                1.0f, dY, out_dim, X, in_dim, 1.0f, G, in_dim);
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

static void layernorm_forward(const float* x, float* y, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    for (int i = 0; i < n; i++) y[i] = (x[i] - mean) * inv_std;
}

static void layernorm_backward(const float* x, const float* dy, float* dx, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    float mean_dy = 0.0f;
    float mean_dy_xhat = 0.0f;
    for (int i = 0; i < n; i++) {
        float xhat = (x[i] - mean) * inv_std;
        mean_dy += dy[i];
        mean_dy_xhat += dy[i] * xhat;
    }
    mean_dy /= (float)n;
    mean_dy_xhat /= (float)n;

    for (int i = 0; i < n; i++) {
        float xhat = (x[i] - mean) * inv_std;
        dx[i] += inv_std * (dy[i] - mean_dy - xhat * mean_dy_xhat);
    }
}

static void norm_forward(const Config* cfg, const float* x, float* y, int n) {
    if (cfg->norm_kind == NORM_LAYER) layernorm_forward(x, y, n);
    else rmsnorm_forward(x, y, n);
}

static void norm_backward(const Config* cfg, const float* x, const float* dy, float* dx, int n) {
    if (cfg->norm_kind == NORM_LAYER) layernorm_backward(x, dy, dx, n);
    else rmsnorm_backward(x, dy, dx, n);
}

static float gelu_forward_scalar(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

static float gelu_backward_scalar(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    const float inv_sqrt_2pi = 0.3989422804014327f;
    return 0.5f * (1.0f + erff(x * inv_sqrt2)) + x * expf(-0.5f * x * x) * inv_sqrt_2pi;
}

static void activation_forward(const Config* cfg, const float* x, float* y, int n) {
    if (cfg->act_kind == ACT_GELU) {
        for (int i = 0; i < n; i++) y[i] = gelu_forward_scalar(x[i]);
    } else if (cfg->act_kind == ACT_RELU) {
        for (int i = 0; i < n; i++) y[i] = fmaxf(0.0f, x[i]);
    } else {
        for (int i = 0; i < n; i++) {
            float r = fmaxf(0.0f, x[i]);
            y[i] = r * r;
        }
    }
}

static void activation_backward(const Config* cfg, const float* x, const float* dy, float* dx, int n) {
    if (cfg->act_kind == ACT_GELU) {
        for (int i = 0; i < n; i++) dx[i] = dy[i] * gelu_backward_scalar(x[i]);
    } else if (cfg->act_kind == ACT_RELU) {
        for (int i = 0; i < n; i++) dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
    } else {
        for (int i = 0; i < n; i++) dx[i] = (x[i] > 0.0f) ? (dy[i] * 2.0f * x[i]) : 0.0f;
    }
}

static void dropout_forward(const Config* cfg, const float* x, float* y, float* mask, int n, bool training) {
    if (!training || cfg->dropout_p <= 0.0f) {
        for (int i = 0; i < n; i++) {
            y[i] = x[i];
            if (mask) mask[i] = 1.0f;
        }
        return;
    }
    float keep = 1.0f - cfg->dropout_p;
    float scale = 1.0f / fmaxf(keep, 1e-6f);
    for (int i = 0; i < n; i++) {
        float m = (rand_uniform() < keep) ? scale : 0.0f;
        if (mask) mask[i] = m;
        y[i] = x[i] * m;
    }
}

static void dropout_backward(const Config* cfg, const float* mask, const float* dy, float* dx, int n) {
    if (cfg->dropout_p <= 0.0f) {
        memcpy(dx, dy, (size_t)n * sizeof(float));
        return;
    }
    for (int i = 0; i < n; i++) dx[i] = dy[i] * mask[i];
}

static const float* lm_head_weight(const Config* cfg, const Weights* w) {
    return cfg->tie_embeddings ? w->wte : w->lm_head;
}

static float* lm_head_grad(const Config* cfg, Grads* g) {
    return cfg->tie_embeddings ? g->wte : g->lm_head;
}

static bool config_matches(const Config* a, const Config* b) {
    return a->n_embd == b->n_embd &&
           a->n_head == b->n_head &&
           a->n_layer == b->n_layer &&
           a->block_size == b->block_size &&
           a->head_dim == b->head_dim &&
           a->vocab_size == b->vocab_size &&
           a->ffn_dim == b->ffn_dim &&
           a->norm_kind == b->norm_kind &&
           a->tie_embeddings == b->tie_embeddings &&
           a->act_kind == b->act_kind &&
           fabsf(a->dropout_p - b->dropout_p) < 1e-8f;
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
    size_t M = (size_t)mlp_dim(cfg);
    size_t lm_head = cfg->tie_embeddings ? 0 : (V * E);
    return V * E + B * E + lm_head + 4 * (L * E * E) + L * M * E + L * E * M;
}

static void alloc_weights(const Config* cfg, Weights* w) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;
    size_t M = (size_t)mlp_dim(cfg);

    w->wte = (float*)malloc(V * E * sizeof(float));
    w->wpe = (float*)malloc(B * E * sizeof(float));
    w->lm_head = (float*)malloc(V * E * sizeof(float));

    w->attn_wq = (float*)malloc(L * E * E * sizeof(float));
    w->attn_wk = (float*)malloc(L * E * E * sizeof(float));
    w->attn_wv = (float*)malloc(L * E * E * sizeof(float));
    w->attn_wo = (float*)malloc(L * E * E * sizeof(float));

    w->mlp_fc1 = (float*)malloc(L * M * E * sizeof(float));
    w->mlp_fc2 = (float*)malloc(L * E * M * sizeof(float));
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
    size_t M = (size_t)mlp_dim(cfg);

    init_mat(w->wte, V * E, 0.02f);
    init_mat(w->wpe, B * E, 0.02f);
    if (cfg->tie_embeddings) memcpy(w->lm_head, w->wte, V * E * sizeof(float));
    else init_mat(w->lm_head, V * E, 0.02f);

    init_mat(w->attn_wq, L * E * E, 0.02f);
    init_mat(w->attn_wk, L * E * E, 0.02f);
    init_mat(w->attn_wv, L * E * E, 0.02f);
    zero_mat(w->attn_wo, L * E * E);

    init_mat(w->mlp_fc1, L * M * E, 0.02f);
    zero_mat(w->mlp_fc2, L * E * M);
}

static void alloc_grads(const Config* cfg, Grads* g) {
    alloc_weights(cfg, (Weights*)g);
}

static void zero_grads(const Config* cfg, Grads* g) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;
    size_t M = (size_t)mlp_dim(cfg);

    zero_mat(g->wte, V * E);
    zero_mat(g->wpe, B * E);
    zero_mat(g->lm_head, V * E);

    zero_mat(g->attn_wq, L * E * E);
    zero_mat(g->attn_wk, L * E * E);
    zero_mat(g->attn_wv, L * E * E);
    zero_mat(g->attn_wo, L * E * E);

    zero_mat(g->mlp_fc1, L * M * E);
    zero_mat(g->mlp_fc2, L * E * M);
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
    int M = mlp_dim(cfg);

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
    c->h1 = (float*)malloc((size_t)L * T * M * sizeof(float));
    c->h2 = (float*)malloc((size_t)L * T * M * sizeof(float));
    c->dropout_mask = (float*)malloc((size_t)L * T * M * sizeof(float));
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
    free(c->dropout_mask);
    free(c->x_out);
    free(c->attn_probs);
}

static inline float* cache_vec(float* base, int layer, int pos, int seq_cap, int width) {
    return base + ((size_t)layer * seq_cap + pos) * width;
}

static inline float* cache_head(float* base, int layer, int pos, int head, int seq_cap, int n_head) {
    return base + (((size_t)layer * seq_cap + pos) * n_head + head) * seq_cap;
}

static float forward_sequence(const Config* cfg, const Weights* w, TrainCache* c, bool training) {
    int T = c->n;
    int E = cfg->n_embd;
    int M = mlp_dim(cfg);
    int H = cfg->n_head;
    int D = cfg->head_dim;
    int L = cfg->n_layer;
    int V = cfg->vocab_size;
    int TC = cfg->block_size;

    float* scratch_probs = (float*)malloc((size_t)V * sizeof(float));
    float* attn_scores = (float*)malloc((size_t)T * T * sizeof(float));
    float total_loss = 0.0f;

    for (int pos = 0; pos < T; pos++) {
        float* es = row(c->embed_sum, E, pos);
        float* x = row(c->x0, E, pos);

        const float* tok = row(w->wte, E, c->in_tokens[pos]);
        const float* pe = row(w->wpe, E, pos);
        for (int i = 0; i < E; i++) es[i] = tok[i] + pe[i];
        norm_forward(cfg, es, x, E);
    }

    float* x_cur = c->x0;
    for (int li = 0; li < L; li++) {
        float* x_in = cache_vec(c->x_in, li, 0, TC, E);
        float* xn_attn = cache_vec(c->xn_attn, li, 0, TC, E);
        float* q = cache_vec(c->q, li, 0, TC, E);
        float* k = cache_vec(c->k, li, 0, TC, E);
        float* v = cache_vec(c->v, li, 0, TC, E);
        float* attn_out = cache_vec(c->attn_out, li, 0, TC, E);
        float* x_after_attn = cache_vec(c->x_after_attn, li, 0, TC, E);
        float* xn_mlp = cache_vec(c->xn_mlp, li, 0, TC, E);
        float* h1 = cache_vec(c->h1, li, 0, TC, M);
        float* h2 = cache_vec(c->h2, li, 0, TC, M);
        float* drop_mask = cache_vec(c->dropout_mask, li, 0, TC, M);
        float* x_out = cache_vec(c->x_out, li, 0, TC, E);

        memcpy(x_in, x_cur, (size_t)T * E * sizeof(float));
        for (int pos = 0; pos < T; pos++) norm_forward(cfg, row(x_in, E, pos), row(xn_attn, E, pos), E);

        const float* Wq = w->attn_wq + (size_t)li * E * E;
        const float* Wk = w->attn_wk + (size_t)li * E * E;
        const float* Wv = w->attn_wv + (size_t)li * E * E;
        const float* Wo = w->attn_wo + (size_t)li * E * E;
        const float* W1 = w->mlp_fc1 + (size_t)li * M * E;
        const float* W2 = w->mlp_fc2 + (size_t)li * E * M;

        linear_seq(Wq, xn_attn, q, T, E, E);
        linear_seq(Wk, xn_attn, k, T, E, E);
        linear_seq(Wv, xn_attn, v, T, E, E);

        memset(attn_out, 0, (size_t)T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            int hs = h * D;
            float scale = 1.0f / sqrtf((float)D);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, T, D,
                        scale, q + hs, E, k + hs, E,
                        0.0f, attn_scores, T);
            for (int pos = 0; pos < T; pos++) {
                float* row_scores = attn_scores + (size_t)pos * T;
                softmax_forward(row_scores, row_scores, pos + 1, 1.0f);
                for (int t = pos + 1; t < T; t++) row_scores[t] = 0.0f;
                memcpy(cache_head(c->attn_probs, li, pos, h, TC, H), row_scores, (size_t)T * sizeof(float));
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, D, T,
                        1.0f, attn_scores, T, v + hs, E,
                        1.0f, attn_out + hs, E);
        }

        linear_seq(Wo, attn_out, x_after_attn, T, E, E);
        for (int i = 0; i < T * E; i++) x_after_attn[i] += x_in[i];

        for (int pos = 0; pos < T; pos++) norm_forward(cfg, row(x_after_attn, E, pos), row(xn_mlp, E, pos), E);
        linear_seq(W1, xn_mlp, h1, T, M, E);
        for (int pos = 0; pos < T; pos++) {
            activation_forward(cfg, row(h1, M, pos), row(h2, M, pos), M);
            dropout_forward(cfg, row(h2, M, pos), row(h2, M, pos), row(drop_mask, M, pos), M, training);
        }
        linear_seq(W2, h2, x_out, T, E, M);
        for (int i = 0; i < T * E; i++) x_out[i] += x_after_attn[i];

        x_cur = x_out;
    }

    memcpy(c->x_final, x_cur, (size_t)T * E * sizeof(float));
    linear_seq(lm_head_weight(cfg, w), c->x_final, c->logits, T, V, E);

    for (int pos = 0; pos < T; pos++) {
        softmax_forward(row(c->logits, V, pos), scratch_probs, V, 1.0f);
        int tgt = c->targets[pos];
        total_loss += -logf(fmaxf(scratch_probs[tgt], 1e-12f));
    }

    free(scratch_probs);
    free(attn_scores);
    return total_loss / (float)T;
}

static void backward_sequence(const Config* cfg, const Weights* w, const TrainCache* c, Grads* g) {
    int T = c->n;
    int E = cfg->n_embd;
    int M = mlp_dim(cfg);
    int H = cfg->n_head;
    int D = cfg->head_dim;
    int L = cfg->n_layer;
    int V = cfg->vocab_size;
    int TC = cfg->block_size;

    float invT = 1.0f / (float)T;

    float* d_logits = (float*)malloc((size_t)T * V * sizeof(float));
    float* d_cur = (float*)calloc((size_t)T * E, sizeof(float));
    float* d_next = (float*)calloc((size_t)T * E, sizeof(float));
    float* d_proj = (float*)malloc((size_t)T * E * sizeof(float));
    float* d_q = (float*)calloc((size_t)T * E, sizeof(float));
    float* dK = (float*)calloc((size_t)T * E, sizeof(float));
    float* dV = (float*)calloc((size_t)T * E, sizeof(float));
    float* d_xn = (float*)malloc((size_t)T * E * sizeof(float));
    float* d_h2 = (float*)malloc((size_t)T * M * sizeof(float));
    float* d_h1 = (float*)malloc((size_t)T * M * sizeof(float));
    float* d_embed = (float*)calloc((size_t)E, sizeof(float));
    float* attn_probs_mat = (float*)malloc((size_t)T * T * sizeof(float));
    float* d_attn_scores = (float*)malloc((size_t)T * T * sizeof(float));
    float* d_a = (float*)malloc((size_t)cfg->block_size * sizeof(float));
    float* d_z = (float*)malloc((size_t)cfg->block_size * sizeof(float));

    for (int pos = 0; pos < T; pos++) {
        float* dlog = row(d_logits, V, pos);
        softmax_forward(row((float*)c->logits, V, pos), dlog, V, 1.0f);
        dlog[c->targets[pos]] -= 1.0f;
        for (int i = 0; i < V; i++) dlog[i] *= invT;
    }
    outer_add_seq(lm_head_grad(cfg, g), d_logits, (float*)c->x_final, T, V, E);
    linear_t_seq(lm_head_weight(cfg, w), d_logits, d_cur, T, V, E);

    for (int li = L - 1; li >= 0; li--) {
        float* x_in = cache_vec((float*)c->x_in, li, 0, TC, E);
        float* xn_attn = cache_vec((float*)c->xn_attn, li, 0, TC, E);
        float* q = cache_vec((float*)c->q, li, 0, TC, E);
        float* k = cache_vec((float*)c->k, li, 0, TC, E);
        float* v = cache_vec((float*)c->v, li, 0, TC, E);
        float* attn_out = cache_vec((float*)c->attn_out, li, 0, TC, E);
        float* x_after_attn = cache_vec((float*)c->x_after_attn, li, 0, TC, E);
        float* xn_mlp = cache_vec((float*)c->xn_mlp, li, 0, TC, E);
        float* h1 = cache_vec((float*)c->h1, li, 0, TC, M);
        float* h2 = cache_vec((float*)c->h2, li, 0, TC, M);
        float* drop_mask = cache_vec((float*)c->dropout_mask, li, 0, TC, M);

        const float* Wq = w->attn_wq + (size_t)li * E * E;
        const float* Wk = w->attn_wk + (size_t)li * E * E;
        const float* Wv = w->attn_wv + (size_t)li * E * E;
        const float* Wo = w->attn_wo + (size_t)li * E * E;
        const float* W1 = w->mlp_fc1 + (size_t)li * M * E;
        const float* W2 = w->mlp_fc2 + (size_t)li * E * M;

        memcpy(d_next, d_cur, (size_t)T * E * sizeof(float));  // residual path from x_out to x_after_attn
        outer_add_seq(g->mlp_fc2 + (size_t)li * E * M, d_cur, h2, T, E, M);
        linear_t_seq(W2, d_cur, d_h2, T, E, M);
        for (int pos = 0; pos < T; pos++) {
            dropout_backward(cfg, row(drop_mask, M, pos), row(d_h2, M, pos), row(d_h2, M, pos), M);
            activation_backward(cfg, row(h1, M, pos), row(d_h2, M, pos), row(d_h1, M, pos), M);
        }
        outer_add_seq(g->mlp_fc1 + (size_t)li * M * E, d_h1, xn_mlp, T, M, E);
        linear_t_seq(W1, d_h1, d_xn, T, M, E);
        for (int pos = 0; pos < T; pos++) {
            norm_backward(cfg, row(x_after_attn, E, pos), row(d_xn, E, pos), row(d_next, E, pos), E);
        }

        outer_add_seq(g->attn_wo + (size_t)li * E * E, d_next, attn_out, T, E, E);
        linear_t_seq(Wo, d_next, d_proj, T, E, E);

        memset(d_q, 0, (size_t)T * E * sizeof(float));
        memset(dK, 0, (size_t)T * E * sizeof(float));
        memset(dV, 0, (size_t)T * E * sizeof(float));
        for (int h = 0; h < H; h++) {
            int hs = h * D;
            float inv_sqrt_d = 1.0f / sqrtf((float)D);
            for (int pos = 0; pos < T; pos++) {
                memcpy(attn_probs_mat + (size_t)pos * T,
                       cache_head((float*)c->attn_probs, li, pos, h, TC, H),
                       (size_t)T * sizeof(float));
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, T, D,
                        1.0f, d_proj + hs, E, v + hs, E,
                        0.0f, d_attn_scores, T);
            for (int pos = 0; pos < T; pos++) {
                const float* probs = attn_probs_mat + (size_t)pos * T;
                float* dA = d_attn_scores + (size_t)pos * T;
                float dot_da_p = 0.0f;
                for (int t = 0; t <= pos; t++) dot_da_p += dA[t] * probs[t];
                for (int t = 0; t <= pos; t++) dA[t] = probs[t] * (dA[t] - dot_da_p);
                for (int t = pos + 1; t < T; t++) dA[t] = 0.0f;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        T, D, T,
                        inv_sqrt_d, d_attn_scores, T, k + hs, E,
                        1.0f, d_q + hs, E);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        T, D, T,
                        inv_sqrt_d, d_attn_scores, T, q + hs, E,
                        1.0f, dK + hs, E);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        T, D, T,
                        1.0f, attn_probs_mat, T, d_proj + hs, E,
                        1.0f, dV + hs, E);
        }

        outer_add_seq(g->attn_wq + (size_t)li * E * E, d_q, xn_attn, T, E, E);
        linear_t_seq(Wq, d_q, d_xn, T, E, E);
        for (int i = 0; i < T * E; i++) d_next[i] += d_xn[i];

        outer_add_seq(g->attn_wk + (size_t)li * E * E, dK, xn_attn, T, E, E);
        linear_t_seq(Wk, dK, d_xn, T, E, E);
        for (int i = 0; i < T * E; i++) d_next[i] += d_xn[i];

        outer_add_seq(g->attn_wv + (size_t)li * E * E, dV, xn_attn, T, E, E);
        linear_t_seq(Wv, dV, d_xn, T, E, E);
        for (int i = 0; i < T * E; i++) d_next[i] += d_xn[i];

        memset(d_cur, 0, (size_t)T * E * sizeof(float));
        for (int pos = 0; pos < T; pos++) {
            norm_backward(cfg, row(x_in, E, pos), row(d_next, E, pos), row(d_cur, E, pos), E);
        }
    }

    for (int pos = 0; pos < T; pos++) {
        for (int i = 0; i < E; i++) d_embed[i] = 0.0f;
        norm_backward(cfg, row((float*)c->embed_sum, E, pos), row(d_cur, E, pos), d_embed, E);

        float* gwte = row(g->wte, E, c->in_tokens[pos]);
        float* gwpe = row(g->wpe, E, pos);
        for (int i = 0; i < E; i++) {
            gwte[i] += d_embed[i];
            gwpe[i] += d_embed[i];
        }
    }
    free(d_embed);

    free(d_logits);
    free(d_cur);
    free(d_next);
    free(d_proj);
    free(d_q);
    free(dK);
    free(dV);
    free(d_xn);
    free(d_h2);
    free(d_h1);
    free(d_a);
    free(d_z);
    free(attn_probs_mat);
    free(d_attn_scores);
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

static void scale_grad_array(float* g, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) g[i] *= scale;
}

static void scale_grads(const Config* cfg, Grads* g, float scale) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;
    size_t M = (size_t)mlp_dim(cfg);

    scale_grad_array(g->wte, V * E, scale);
    scale_grad_array(g->wpe, B * E, scale);
    scale_grad_array(g->lm_head, V * E, scale);
    scale_grad_array(g->attn_wq, L * E * E, scale);
    scale_grad_array(g->attn_wk, L * E * E, scale);
    scale_grad_array(g->attn_wv, L * E * E, scale);
    scale_grad_array(g->attn_wo, L * E * E, scale);
    scale_grad_array(g->mlp_fc1, L * M * E, scale);
    scale_grad_array(g->mlp_fc2, L * E * M, scale);
}

static void adam_update(const Config* cfg, Weights* w, const Grads* g, AdamBuf* m1, AdamBuf* m2,
                        float lr_t, float beta1, float beta2, float eps, int step1) {
    size_t V = (size_t)cfg->vocab_size;
    size_t E = (size_t)cfg->n_embd;
    size_t L = (size_t)cfg->n_layer;
    size_t B = (size_t)cfg->block_size;
    size_t M = (size_t)mlp_dim(cfg);

    adam_update_array(w->wte, g->wte, m1->wte, m2->wte, V * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->wpe, g->wpe, m1->wpe, m2->wpe, B * E, lr_t, beta1, beta2, eps, step1);
    if (!cfg->tie_embeddings) {
        adam_update_array(w->lm_head, g->lm_head, m1->lm_head, m2->lm_head, V * E, lr_t, beta1, beta2, eps, step1);
    }

    adam_update_array(w->attn_wq, g->attn_wq, m1->attn_wq, m2->attn_wq, L * E * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->attn_wk, g->attn_wk, m1->attn_wk, m2->attn_wk, L * E * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->attn_wv, g->attn_wv, m1->attn_wv, m2->attn_wv, L * E * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->attn_wo, g->attn_wo, m1->attn_wo, m2->attn_wo, L * E * E, lr_t, beta1, beta2, eps, step1);

    adam_update_array(w->mlp_fc1, g->mlp_fc1, m1->mlp_fc1, m2->mlp_fc1, L * M * E, lr_t, beta1, beta2, eps, step1);
    adam_update_array(w->mlp_fc2, g->mlp_fc2, m1->mlp_fc2, m2->mlp_fc2, L * E * M, lr_t, beta1, beta2, eps, step1);
}

static void gpt_infer_step(const Config* cfg, const Weights* w,
                           float* kcache, float* vcache,
                           int token_id, int pos_id, float* logits) {
    int E = cfg->n_embd;
    int H = cfg->n_head;
    int D = cfg->head_dim;
    int L = cfg->n_layer;
    int B = cfg->block_size;
    int M = mlp_dim(cfg);

    float* x = (float*)malloc((size_t)E * sizeof(float));
    float* tmp = (float*)malloc((size_t)E * sizeof(float));
    float* xn = (float*)malloc((size_t)E * sizeof(float));
    float* q = (float*)malloc((size_t)E * sizeof(float));
    float* k = (float*)malloc((size_t)E * sizeof(float));
    float* v = (float*)malloc((size_t)E * sizeof(float));
    float* attn_out = (float*)malloc((size_t)E * sizeof(float));
    float* mlp = (float*)malloc((size_t)M * sizeof(float));

    const float* tok = row(w->wte, E, token_id);
    const float* pe = row(w->wpe, E, pos_id);
    for (int i = 0; i < E; i++) tmp[i] = tok[i] + pe[i];
    norm_forward(cfg, tmp, x, E);

    for (int li = 0; li < L; li++) {
        memcpy(tmp, x, (size_t)E * sizeof(float));
        norm_forward(cfg, x, xn, E);

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
        norm_forward(cfg, x, xn, E);

        const float* W1 = w->mlp_fc1 + (size_t)li * M * E;
        const float* W2 = w->mlp_fc2 + (size_t)li * E * M;
        linear(W1, xn, mlp, M, E);
        activation_forward(cfg, mlp, mlp, M);
        linear(W2, mlp, x, E, M);
        for (int i = 0; i < E; i++) x[i] += tmp[i];
    }

    linear(lm_head_weight(cfg, w), x, logits, cfg->vocab_size, E);

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
        s += forward_sequence(cfg, w, cache, false);
    }
    return s / (float)eval_iters;
}

static float eval_token_loss(const Config* cfg, const Weights* w, TrainCache* cache,
                             const uint16_t* tokens, size_t n_tokens, int eval_iters) {
    if (n_tokens <= 1) return NAN;
    float s = 0.0f;
    for (int i = 0; i < eval_iters; i++) {
        prepare_token_example(tokens, n_tokens, cfg->block_size, cache);
        s += forward_sequence(cfg, w, cache, false);
    }
    return s / (float)eval_iters;
}

static int env_int_or(const char* key, int fallback) {
    const char* v = getenv(key);
    return (v && v[0] != '\0') ? atoi(v) : fallback;
}

static float env_float_or(const char* key, float fallback) {
    const char* v = getenv(key);
    return (v && v[0] != '\0') ? atof(v) : fallback;
}

static bool env_equals(const char* key, const char* expected) {
    const char* v = getenv(key);
    return v && strcmp(v, expected) == 0;
}

static bool save_checkpoint(const char* path, const Config* cfg, const Weights* w) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    const uint32_t magic = 0x4D475043;  // MGPC
    const uint32_t version = 4;
    if (fwrite(&magic, sizeof(magic), 1, f) != 1) goto fail;
    if (fwrite(&version, sizeof(version), 1, f) != 1) goto fail;
    if (fwrite(cfg, sizeof(*cfg), 1, f) != 1) goto fail;

    size_t V = (size_t)cfg->vocab_size, E = (size_t)cfg->n_embd, L = (size_t)cfg->n_layer, B = (size_t)cfg->block_size, M = (size_t)mlp_dim(cfg);
    if (fwrite(w->wte, sizeof(float), V * E, f) != V * E) goto fail;
    if (fwrite(w->wpe, sizeof(float), B * E, f) != B * E) goto fail;
    const float* lm_head = cfg->tie_embeddings ? w->wte : w->lm_head;
    if (fwrite(lm_head, sizeof(float), V * E, f) != V * E) goto fail;
    if (fwrite(w->attn_wq, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->attn_wk, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->attn_wv, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->attn_wo, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fwrite(w->mlp_fc1, sizeof(float), L * M * E, f) != L * M * E) goto fail;
    if (fwrite(w->mlp_fc2, sizeof(float), L * E * M, f) != L * E * M) goto fail;
    fclose(f);
    return true;
fail:
    fclose(f);
    return false;
}

static bool load_checkpoint_config(const char* path, Config* out_cfg) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0;
    bool ok = false;
    if (fread(&magic, sizeof(magic), 1, f) != 1) goto done;
    if (fread(&version, sizeof(version), 1, f) != 1) goto done;
    if (magic != 0x4D475043) goto done;
    if (version == 1) {
        ConfigV1 old = {0};
        if (fread(&old, sizeof(old), 1, f) != 1) goto done;
        *out_cfg = (Config){
            .n_embd = old.n_embd,
            .n_head = old.n_head,
            .n_layer = old.n_layer,
            .block_size = old.block_size,
            .head_dim = old.head_dim,
            .vocab_size = old.vocab_size,
            .ffn_dim = 4 * old.n_embd,
            .norm_kind = NORM_RMS,
            .tie_embeddings = 0,
            .act_kind = ACT_RELU2,
            .dropout_p = 0.0f,
        };
        ok = true;
    } else if (version == 2) {
        ConfigV2 old = {0};
        if (fread(&old, sizeof(old), 1, f) != 1) goto done;
        *out_cfg = (Config){
            .n_embd = old.n_embd,
            .n_head = old.n_head,
            .n_layer = old.n_layer,
            .block_size = old.block_size,
            .head_dim = old.head_dim,
            .vocab_size = old.vocab_size,
            .ffn_dim = 4 * old.n_embd,
            .norm_kind = old.norm_kind,
            .tie_embeddings = old.tie_embeddings,
            .act_kind = ACT_RELU2,
            .dropout_p = 0.0f,
        };
        ok = true;
    } else if (version == 3) {
        ConfigV3 old = {0};
        if (fread(&old, sizeof(old), 1, f) != 1) goto done;
        *out_cfg = (Config){
            .n_embd = old.n_embd,
            .n_head = old.n_head,
            .n_layer = old.n_layer,
            .block_size = old.block_size,
            .head_dim = old.head_dim,
            .vocab_size = old.vocab_size,
            .ffn_dim = 4 * old.n_embd,
            .norm_kind = old.norm_kind,
            .tie_embeddings = old.tie_embeddings,
            .act_kind = old.act_kind,
            .dropout_p = old.dropout_p,
        };
        ok = true;
    } else if (version == 4) {
        if (fread(out_cfg, sizeof(*out_cfg), 1, f) != 1) goto done;
        ok = true;
    }
done:
    fclose(f);
    return ok;
}

static bool load_checkpoint(const char* path, const Config* cfg, Weights* w) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0;
    Config ck = {0};
    if (fread(&magic, sizeof(magic), 1, f) != 1) goto fail;
    if (fread(&version, sizeof(version), 1, f) != 1) goto fail;
    if (magic != 0x4D475043) goto fail;
    if (version == 1) {
        ConfigV1 old = {0};
        if (fread(&old, sizeof(old), 1, f) != 1) goto fail;
        ck = (Config){
            .n_embd = old.n_embd,
            .n_head = old.n_head,
            .n_layer = old.n_layer,
            .block_size = old.block_size,
            .head_dim = old.head_dim,
            .vocab_size = old.vocab_size,
            .ffn_dim = 4 * old.n_embd,
            .norm_kind = NORM_RMS,
            .tie_embeddings = 0,
            .act_kind = ACT_RELU2,
            .dropout_p = 0.0f,
        };
    } else if (version == 2) {
        ConfigV2 old = {0};
        if (fread(&old, sizeof(old), 1, f) != 1) goto fail;
        ck = (Config){
            .n_embd = old.n_embd,
            .n_head = old.n_head,
            .n_layer = old.n_layer,
            .block_size = old.block_size,
            .head_dim = old.head_dim,
            .vocab_size = old.vocab_size,
            .ffn_dim = 4 * old.n_embd,
            .norm_kind = old.norm_kind,
            .tie_embeddings = old.tie_embeddings,
            .act_kind = ACT_RELU2,
            .dropout_p = 0.0f,
        };
    } else if (version == 3) {
        ConfigV3 old = {0};
        if (fread(&old, sizeof(old), 1, f) != 1) goto fail;
        ck = (Config){
            .n_embd = old.n_embd,
            .n_head = old.n_head,
            .n_layer = old.n_layer,
            .block_size = old.block_size,
            .head_dim = old.head_dim,
            .vocab_size = old.vocab_size,
            .ffn_dim = 4 * old.n_embd,
            .norm_kind = old.norm_kind,
            .tie_embeddings = old.tie_embeddings,
            .act_kind = old.act_kind,
            .dropout_p = old.dropout_p,
        };
    } else if (version == 4) {
        if (fread(&ck, sizeof(ck), 1, f) != 1) goto fail;
    } else {
        goto fail;
    }
    if (!config_matches(&ck, cfg)) goto fail;

    size_t V = (size_t)cfg->vocab_size, E = (size_t)cfg->n_embd, L = (size_t)cfg->n_layer, B = (size_t)cfg->block_size, M = (size_t)mlp_dim(cfg);
    if (fread(w->wte, sizeof(float), V * E, f) != V * E) goto fail;
    if (fread(w->wpe, sizeof(float), B * E, f) != B * E) goto fail;
    if (fread(w->lm_head, sizeof(float), V * E, f) != V * E) goto fail;
    if (fread(w->attn_wq, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->attn_wk, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->attn_wv, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->attn_wo, sizeof(float), L * E * E, f) != L * E * E) goto fail;
    if (fread(w->mlp_fc1, sizeof(float), L * M * E, f) != L * M * E) goto fail;
    if (fread(w->mlp_fc2, sizeof(float), L * E * M, f) != L * E * M) goto fail;
    fclose(f);
    return true;
fail:
    fclose(f);
    return false;
}

static bool chat_with_prompt(const Config* cfg, const Weights* w, const char* prompt, float temperature,
                             const char* id_to_char, const int* char_to_id, int BOS) {
    char raw_prefix[1024];
    char prefix[1024];
    format_chat_prompt(prompt, raw_prefix, sizeof(raw_prefix));
    truncate_to_tail(raw_prefix, cfg->block_size - 1, prefix, sizeof(prefix));

    for (size_t i = 0; prefix[i] != '\0'; i++) {
        if (char_to_id[(unsigned char)prefix[i]] < 0) {
            fprintf(stderr, "error: prompt contains unsupported character '%c'\n", prefix[i]);
            return false;
        }
    }

    float* logits = (float*)malloc((size_t)cfg->vocab_size * sizeof(float));
    float* probs = (float*)malloc((size_t)cfg->vocab_size * sizeof(float));
    float* kcache = (float*)calloc((size_t)cfg->n_layer * cfg->block_size * cfg->n_embd, sizeof(float));
    float* vcache = (float*)calloc((size_t)cfg->n_layer * cfg->block_size * cfg->n_embd, sizeof(float));
    char generated[4096];
    generated[0] = '\0';

    int token = BOS;
    int pos = 0;
    for (; prefix[pos] != '\0' && pos < cfg->block_size - 1; pos++) {
        gpt_infer_step(cfg, w, kcache, vcache, token, pos, logits);
        token = char_to_id[(unsigned char)prefix[pos]];
    }

    while (pos < cfg->block_size - 1) {
        gpt_infer_step(cfg, w, kcache, vcache, token, pos, logits);
        softmax_forward(logits, probs, cfg->vocab_size, temperature);
        token = sample_from_probs(probs, cfg->vocab_size);
        if (token == BOS) break;

        size_t len = strlen(generated);
        if (len + 2 >= sizeof(generated)) break;
        generated[len] = id_to_char[token];
        generated[len + 1] = '\0';

        char* end_tag = strstr(generated, kChatEndTag);
        if (end_tag) {
            *end_tag = '\0';
            break;
        }
        pos++;
    }

    printf("you> %s\n", prompt);
    printf("guppy> %s\n", generated);

    free(logits);
    free(probs);
    free(kcache);
    free(vcache);
    return true;
}

static bool token_chat_with_prompt(const Config* cfg, const Weights* w,
                                   const int* prompt_tokens, int n_prompt, int stop_token,
                                   float temperature, int max_new_tokens) {
    if (n_prompt <= 0) {
        fprintf(stderr, "error: token prompt is empty\n");
        return false;
    }
    if (max_new_tokens <= 0) {
        fprintf(stderr, "error: max_new_tokens must be positive\n");
        return false;
    }

    int usable = n_prompt;
    const int* prompt_tail = prompt_tokens;
    if (usable > cfg->block_size - 1) {
        prompt_tail += usable - (cfg->block_size - 1);
        usable = cfg->block_size - 1;
    }

    float* logits = (float*)malloc((size_t)cfg->vocab_size * sizeof(float));
    float* probs = (float*)malloc((size_t)cfg->vocab_size * sizeof(float));
    float* kcache = (float*)calloc((size_t)cfg->n_layer * cfg->block_size * cfg->n_embd, sizeof(float));
    float* vcache = (float*)calloc((size_t)cfg->n_layer * cfg->block_size * cfg->n_embd, sizeof(float));
    int* generated = (int*)malloc((size_t)max_new_tokens * sizeof(int));
    if (!logits || !probs || !kcache || !vcache || !generated) {
        fprintf(stderr, "error: allocation failed in token chat\n");
        free(logits);
        free(probs);
        free(kcache);
        free(vcache);
        free(generated);
        return false;
    }

    int token = prompt_tail[0];
    int pos = 0;
    for (int i = 1; i < usable; i++) {
        gpt_infer_step(cfg, w, kcache, vcache, token, pos, logits);
        token = prompt_tail[i];
        pos++;
    }

    int n_generated = 0;
    while (pos < cfg->block_size - 1 && n_generated < max_new_tokens) {
        gpt_infer_step(cfg, w, kcache, vcache, token, pos, logits);
        softmax_forward(logits, probs, cfg->vocab_size, temperature);
        token = sample_from_probs(probs, cfg->vocab_size);
        if (stop_token >= 0 && token == stop_token) break;
        generated[n_generated++] = token;
        pos++;
    }

    printf("guppy_token_ids>");
    for (int i = 0; i < n_generated; i++) printf("%s%d", i == 0 ? " " : " ", generated[i]);
    putchar('\n');

    free(logits);
    free(probs);
    free(kcache);
    free(vcache);
    free(generated);
    return true;
}

static int run_bpe_chat_script(const char* binary_path, const char* prompt, const char* dataset_path,
                               const char* ckpt_path, float temperature, int max_new_tokens) {
    char data_dir[1024];
    char temp_buf[64];
    char max_buf[64];
    infer_parent_dir(dataset_path, data_dir, sizeof(data_dir));
    snprintf(temp_buf, sizeof(temp_buf), "%.4f", temperature);
    snprintf(max_buf, sizeof(max_buf), "%d", max_new_tokens);

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "error: failed to start python chat helper\n");
        return 1;
    }
    if (pid == 0) {
        execlp("python3", "python3",
               "scripts/chat_guppy_bpe.py",
               prompt,
               "--data-dir", data_dir,
               "--ckpt", ckpt_path,
               "--binary", binary_path,
               "--temperature", temp_buf,
               "--max-new-tokens", max_buf,
               (char*)NULL);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        fprintf(stderr, "error: failed waiting for python chat helper\n");
        return 1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 1;
}

int main(int argc, char** argv) {
    srand(42);  // align with Python gist

    if (argc > 1 && strcmp(argv[1], "guppy-data") == 0) {
        const char* out_path = (argc > 2) ? argv[2] : "guppy_input.txt";
        int n_samples = (argc > 3) ? atoi(argv[3]) : 60000;
        if (n_samples <= 0) {
            fprintf(stderr, "error: sample count must be positive\n");
            return 1;
        }
        if (!write_guppy_dataset(out_path, n_samples)) {
            fprintf(stderr, "error: failed to write dataset to %s\n", out_path);
            return 1;
        }
        printf("wrote %d guppy chat samples to %s\n", n_samples, out_path);
        return 0;
    }

    if (argc > 1 && strcmp(argv[1], "chat") == 0) {
        const char* prompt = (argc > 2) ? argv[2] : "hi guppy";
        const char* dataset_path = (argc > 3) ? argv[3] : "guppy_input.txt";
        const char* ckpt_path = (argc > 4) ? argv[4] : "ckpt_best.bin";
        float temperature = (argc > 5) ? atof(argv[5]) : 0.7f;

        if (path_has_suffix(dataset_path, ".bin")) {
            return run_bpe_chat_script(argv[0], prompt, dataset_path, ckpt_path, temperature, 64);
        }

        char** docs = NULL;
        int n_docs = 0;
        if (!load_lines(dataset_path, &docs, &n_docs)) {
            fprintf(stderr, "error: failed to load dataset file %s\n", dataset_path);
            return 1;
        }

        char id_to_char[256] = {0};
        int char_to_id[256];
        for (int i = 0; i < 256; i++) char_to_id[i] = -1;
        int vocab_size = 0;
        build_vocab(docs, n_docs, id_to_char, &vocab_size, char_to_id);
        int BOS = vocab_size - 1;

        Config cfg = {0};
        if (!load_checkpoint_config(ckpt_path, &cfg)) {
            fprintf(stderr, "error: failed to read checkpoint config from %s\n", ckpt_path);
            free_lines(docs, n_docs);
            return 1;
        }
        cfg.vocab_size = vocab_size;

        Weights w = {0};
        alloc_weights(&cfg, &w);
        if (!load_checkpoint(ckpt_path, &cfg, &w)) {
            fprintf(stderr, "error: failed to load checkpoint %s\n", ckpt_path);
            free_weights(&w);
            free_lines(docs, n_docs);
            return 1;
        }

        bool ok = chat_with_prompt(&cfg, &w, prompt, temperature, id_to_char, char_to_id, BOS);
        free_weights(&w);
        free_lines(docs, n_docs);
        return ok ? 0 : 1;
    }

    if (argc > 1 && strcmp(argv[1], "token-chat") == 0) {
        const char* prompt_ids_text = (argc > 2) ? argv[2] : "";
        const char* dataset_path = (argc > 3) ? argv[3] : "data/guppy_bpe/train.bin";
        const char* ckpt_path = (argc > 4) ? argv[4] : "ckpt_best.bin";
        float temperature = (argc > 5) ? atof(argv[5]) : 0.7f;
        int max_new_tokens = (argc > 6) ? atoi(argv[6]) : 64;

        if (!path_has_suffix(dataset_path, ".bin")) {
            fprintf(stderr, "error: token-chat expects a train.bin dataset path\n");
            return 1;
        }

        TokenCorpus token_corpus = {0};
        if (!load_token_corpus(dataset_path, &token_corpus)) {
            fprintf(stderr, "error: failed to load token corpus from %s\n", dataset_path);
            return 1;
        }

        Config cfg = {0};
        if (!load_checkpoint_config(ckpt_path, &cfg)) {
            fprintf(stderr, "error: failed to read checkpoint config from %s\n", ckpt_path);
            free_token_corpus(&token_corpus);
            return 1;
        }
        cfg.vocab_size = token_corpus.vocab_size;

        int max_prompt_tokens = (int)strlen(prompt_ids_text) + 1;
        int* prompt_ids = (int*)malloc((size_t)max_prompt_tokens * sizeof(int));
        if (!prompt_ids) {
            fprintf(stderr, "error: failed to allocate prompt buffer\n");
            free_token_corpus(&token_corpus);
            return 1;
        }
        int n_prompt = parse_token_id_list(prompt_ids_text, cfg.vocab_size, prompt_ids, max_prompt_tokens);
        if (n_prompt <= 0) {
            fprintf(stderr, "error: failed to parse prompt token ids\n");
            free(prompt_ids);
            free_token_corpus(&token_corpus);
            return 1;
        }

        Weights w = {0};
        alloc_weights(&cfg, &w);
        if (!load_checkpoint(ckpt_path, &cfg, &w)) {
            fprintf(stderr, "error: failed to load checkpoint %s\n", ckpt_path);
            free_weights(&w);
            free(prompt_ids);
            free_token_corpus(&token_corpus);
            return 1;
        }

        bool ok = token_chat_with_prompt(&cfg, &w, prompt_ids, n_prompt, token_corpus.end_id, temperature, max_new_tokens);
        free_weights(&w);
        free(prompt_ids);
        free_token_corpus(&token_corpus);
        return ok ? 0 : 1;
    }

    const char* dataset_path = "input.txt";
    if (argc > 11) dataset_path = argv[11];
    bool token_mode = path_has_suffix(dataset_path, ".bin");

    char** docs = NULL;
    int n_docs = 0;
    TokenCorpus token_corpus = {0};

    if (!token_mode && strcmp(dataset_path, "input.txt") == 0 && !ensure_input_file()) {
        fprintf(stderr, "warning: input.txt missing and auto-download failed, using tiny fallback corpus\n");
    }
    if (token_mode) {
        if (!load_token_corpus(dataset_path, &token_corpus)) {
            fprintf(stderr, "error: failed to load token corpus from %s\n", dataset_path);
            return 1;
        }
    } else if (!load_lines(dataset_path, &docs, &n_docs)) {
        static const char* fallback[] = {"anna", "bob", "carol", "david", "emma", "frank"};
        n_docs = (int)(sizeof(fallback) / sizeof(fallback[0]));
        docs = (char**)malloc((size_t)n_docs * sizeof(char*));
        for (int i = 0; i < n_docs; i++) docs[i] = strdup(fallback[i]);
    }
    if (!token_mode) shuffle_lines(docs, n_docs);

    char id_to_char[256] = {0};
    int char_to_id[256];
    for (int i = 0; i < 256; i++) char_to_id[i] = -1;

    int vocab_size = 0;
    int BOS = -1;
    if (!token_mode) {
        build_vocab(docs, n_docs, id_to_char, &vocab_size, char_to_id);
        BOS = vocab_size - 1;
    } else {
        vocab_size = token_corpus.vocab_size;
    }

    int num_steps = 500;
    float temperature = 0.5f;
    int num_samples = 20;
    int eval_interval = 100;
    int eval_iters = 100;
    float learning_rate = 3e-3f;
    int n_embd = 32;
    int n_head = 4;
    int n_layer = 2;
    int block_size = 16;
    int ffn_dim = 4 * n_embd;
    int norm_kind = NORM_RMS;
    int tie_embeddings = 0;
    int act_kind = ACT_RELU2;
    float dropout_p = 0.0f;
    int batch_size = env_int_or("MICROGPT_BATCH_SIZE", 1);
    if (argc > 1) num_steps = atoi(argv[1]);
    if (argc > 2) temperature = atof(argv[2]);
    if (argc > 3) num_samples = atoi(argv[3]);
    if (argc > 4) eval_interval = atoi(argv[4]);
    if (argc > 5) eval_iters = atoi(argv[5]);
    if (argc > 6) n_embd = atoi(argv[6]);
    if (argc > 7) n_head = atoi(argv[7]);
    if (argc > 8) n_layer = atoi(argv[8]);
    if (argc > 9) block_size = atoi(argv[9]);
    if (argc > 10) learning_rate = atof(argv[10]);
    const char* chat_prompt = (argc > 12) ? argv[12] : NULL;
    ffn_dim = 4 * n_embd;

    if (token_mode) {
        norm_kind = NORM_LAYER;
        tie_embeddings = 1;
        act_kind = ACT_RELU;
        dropout_p = 0.1f;
        if (block_size < 128) block_size = 128;
        if (learning_rate > 1e-3f) learning_rate = 1e-3f;
    } else if (dataset_looks_like_chat(docs, n_docs)) {
        norm_kind = NORM_LAYER;
        tie_embeddings = 1;
        act_kind = ACT_RELU;
        dropout_p = 0.1f;
        if (block_size < 128) block_size = 128;
        if (learning_rate > 1e-3f) learning_rate = 1e-3f;
    }

    if (env_equals("MICROGPT_ACT", "gelu")) act_kind = ACT_GELU;
    if (env_equals("MICROGPT_ACT", "relu2")) act_kind = ACT_RELU2;
    if (env_equals("MICROGPT_ACT", "relu")) act_kind = ACT_RELU;
    dropout_p = env_float_or("MICROGPT_DROPOUT", dropout_p);
    ffn_dim = env_int_or("MICROGPT_FFN_DIM", ffn_dim);
    if (token_mode || dataset_looks_like_chat(docs, n_docs)) {
        ffn_dim = env_int_or("MICROGPT_FFN_DIM", 2 * n_embd);
    }

    if (n_embd <= 0 || n_head <= 0 || n_layer <= 0 || block_size <= 0 || ffn_dim <= 0) {
        fprintf(stderr, "error: n_embd, n_head, n_layer, block_size, ffn_dim must be positive\n");
        return 1;
    }
    if (learning_rate <= 0.0f) {
        fprintf(stderr, "error: learning_rate must be positive\n");
        return 1;
    }
    if ((n_embd % n_head) != 0) {
        fprintf(stderr, "error: n_embd (%d) must be divisible by n_head (%d)\n", n_embd, n_head);
        return 1;
    }
    if (batch_size <= 0) {
        fprintf(stderr, "error: MICROGPT_BATCH_SIZE must be positive\n");
        return 1;
    }
    if (dropout_p < 0.0f || dropout_p >= 1.0f) {
        fprintf(stderr, "error: MICROGPT_DROPOUT must be in [0, 1)\n");
        return 1;
    }

    Config cfg = {
        .n_embd = n_embd,
        .n_head = n_head,
        .n_layer = n_layer,
        .block_size = block_size,
        .head_dim = n_embd / n_head,
        .vocab_size = vocab_size,
        .ffn_dim = ffn_dim,
        .norm_kind = norm_kind,
        .tie_embeddings = tie_embeddings,
        .act_kind = act_kind,
        .dropout_p = dropout_p,
    };

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

    printf("dataset: %s\n", dataset_path);
    if (token_mode) {
        printf("token mode: train_tokens=%zu val_tokens=%zu\n",
               token_corpus.n_train_tokens, token_corpus.n_val_tokens);
    } else {
        printf("num docs: %d\n", n_docs);
    }
    printf("vocab size: %d\n", cfg.vocab_size);
    printf("model: n_embd=%d n_head=%d n_layer=%d block_size=%d ffn_dim=%d lr=%.6f batch=%d norm=%s tied_embeddings=%s act=%s dropout=%.2f\n",
           cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block_size, cfg.ffn_dim, learning_rate,
           batch_size, norm_kind_name(cfg.norm_kind), cfg.tie_embeddings ? "yes" : "no",
           act_kind_name(cfg.act_kind), cfg.dropout_p);
    printf("num params: %zu\n", param_count(&cfg));

    float beta1 = 0.9f, beta2 = 0.95f, eps_adam = 1e-8f;
    int n_val = 0;
    int n_train = 0;
    if (!token_mode) {
        n_val = n_docs / 10;
        if (n_docs > 1 && n_val < 1) n_val = 1;
        if (n_docs > 1 && n_val >= n_docs) n_val = n_docs - 1;
        if (n_docs <= 1) n_val = 0;
        n_train = n_docs - n_val;
        printf("train docs: %d | val docs: %d\n", n_train, n_val);
    }
    int* toks = (int*)malloc((size_t)(cfg.block_size + 2) * sizeof(int));

    float* losses = (float*)malloc((size_t)num_steps * sizeof(float));
    float best_val = INFINITY;
    const char* best_ckpt = "ckpt_best.bin";
    FILE* ef = fopen("eval.csv", "w");
    if (ef) fprintf(ef, "step,train_loss,val_loss,lr\n");
    struct timespec t0 = {0}, t1 = {0};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int step = 0; step < num_steps; step++) {
        zero_grads(&cfg, &g);
        float loss = 0.0f;
        for (int bi = 0; bi < batch_size; bi++) {
            if (token_mode) prepare_token_example(token_corpus.train_tokens, token_corpus.n_train_tokens, cfg.block_size, &cache);
            else {
                int idx = rand() % n_train;
                prepare_example(docs[idx], BOS, char_to_id, cfg.block_size, toks, &cache);
            }
            loss += forward_sequence(&cfg, &w, &cache, true);
            backward_sequence(&cfg, &w, &cache, &g);
        }
        loss /= (float)batch_size;
        scale_grads(&cfg, &g, 1.0f / (float)batch_size);

        float lr_t = learning_rate * 0.5f * (1.0f + cosf((float)M_PI * (float)step / (float)num_steps));
        adam_update(&cfg, &w, &g, &m1, &m2, lr_t, beta1, beta2, eps_adam, step + 1);

        losses[step] = loss;
        printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss);

        bool do_eval = ((step + 1) % eval_interval == 0) || (step + 1 == num_steps);
        if (do_eval) {
            float train_eval = token_mode
                ? eval_token_loss(&cfg, &w, &cache, token_corpus.train_tokens, token_corpus.n_train_tokens, eval_iters)
                : eval_split_loss(&cfg, &w, &cache, docs, 0, n_train, BOS, char_to_id, toks, eval_iters);
            float val_eval = token_mode
                ? eval_token_loss(&cfg, &w, &cache, token_corpus.val_tokens, token_corpus.n_val_tokens, eval_iters)
                : ((n_val > 0) ? eval_split_loss(&cfg, &w, &cache, docs, n_train, n_val, BOS, char_to_id, toks, eval_iters) : NAN);
            printf("eval step %4d | train %.4f | val %.4f | lr %.6f\n", step + 1, train_eval, val_eval, lr_t);
            if (ef) fprintf(ef, "%d,%.8f,%.8f,%.8f\n", step + 1, train_eval, val_eval, lr_t);

            float metric = token_mode ? val_eval : ((n_val > 0) ? val_eval : train_eval);
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
    if (token_mode) {
        printf("token-mode training complete.\n");
        printf("chat with: ./microgpt_mac chat \"tell me a joke\" %s ckpt_best.bin 0.7\n", dataset_path);
    } else if (chat_prompt && chat_prompt[0] != '\0') {
        chat_with_prompt(&cfg, &w, chat_prompt, temperature, id_to_char, char_to_id, BOS);
    } else {
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
    }
    free(toks);

    free_train_cache(&cache);
    free_adam(&m1);
    free_adam(&m2);
    free_grads(&g);
    free_weights(&w);
    if (token_mode) free_token_corpus(&token_corpus);
    else free_lines(docs, n_docs);
    return 0;
}
