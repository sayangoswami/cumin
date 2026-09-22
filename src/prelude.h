//
// Created by Sayan Goswami on 22.09.2026.
//

#ifndef CUMIN_PRELUDE_H
#define CUMIN_PRELUDE_H

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define RESET "\x1B[0m"

static char *time_str(){
    time_t rawtime;
    struct tm * timeinfo;
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    static char buf[9];
    strftime (buf, 9,"%T",timeinfo);
    return buf;
}

#define log_error(fmt, ...) do { \
    fprintf(stderr, "[%s]" RED "ERROR: " fmt RESET " at %s:%i\n",       \
            time_str(), ##__VA_ARGS__, __FILE__, __LINE__);     \
    exit(1);                                                                      \
} while(0)

#define log_info(fmt, ...) do { \
    fprintf(stderr, "[%s]" GRN "INFO: " fmt RESET "\n", time_str(), ##__VA_ARGS__); \
} while(0)

#define log_warn(fmt, ...) do { \
    fprintf(stderr, "[%s]" YEL "WARN: " fmt RESET " at %s:%i\n",       \
            time_str(), ##__VA_ARGS__, __FILE__, __LINE__);     \
} while(0)

#define expect(expression) if (!(expression)) log_error("Expected " #expression "")

/// aliases and typedefs
typedef uint8_t u1;
typedef uint16_t u2;
typedef uint32_t u4;
typedef uint64_t u8;
typedef int32_t i4;
typedef int64_t i8;

/** memory size units, e.g. `64 MiB` */
#define KiB <<10u
#define MiB <<20u
#define GiB <<30u

/** generic macros */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/** I/O utilities -- fixed-width dump/load of scalars and parlay::sequence-like containers */

template <class T>
static void dump_values(std::ostream &f, T& var) {
    f.write(reinterpret_cast<const char*>(&var), sizeof(var));
}

template <class T, typename... Args>
static void dump_values(std::ostream &f, T& var, Args... args) {
    f.write(reinterpret_cast<const char*>(&var), sizeof(var));
    dump_values(f, args...);
}

template <class T>
static void load_values(std::istream &f, T *p_var) {
    f.read(reinterpret_cast<char*>(p_var), sizeof(*p_var));
}

template <class T, typename... Args>
static void load_values(std::istream &f, T *p_var, Args... args) {
    f.read(reinterpret_cast<char*>(p_var), sizeof(*p_var));
    load_values(f, args...);
}

template <class Seq>
static inline void dump_seq(std::ostream &f, const Seq &seq) {
    size_t n = seq.size();
    dump_values(f, n);
    f.write(reinterpret_cast<const char*>(seq.data()), (std::streamsize)(n * sizeof(seq[0])));
}

template <class Seq>
static inline void load_seq(std::istream &f, Seq &seq) {
    size_t n = 0;
    load_values(f, &n);
    seq.resize(n);
    f.read(reinterpret_cast<char*>(seq.data()), (std::streamsize)(n * sizeof(seq[0])));
}

static inline void dump_string(std::ostream &f, const std::string &s) {
    size_t n = s.size();
    dump_values(f, n);
    f.write(s.data(), (std::streamsize)n);
}

static inline void load_string(std::istream &f, std::string &s) {
    size_t n = 0;
    load_values(f, &n);
    s.resize(n);
    f.read(s.data(), (std::streamsize)n);
}

static inline void dump_strings(std::ostream &f, const std::vector<std::string> &v) {
    size_t n = v.size();
    dump_values(f, n);
    for (auto &s: v) dump_string(f, s);
}

static inline void load_strings(std::istream &f, std::vector<std::string> &v) {
    size_t n = 0;
    load_values(f, &n);
    v.resize(n);
    for (auto &s: v) load_string(f, s);
}

#endif //CUMIN_PRELUDE_H
