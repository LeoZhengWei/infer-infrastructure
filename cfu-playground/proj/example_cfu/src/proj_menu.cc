/*
 * Copyright 2021 The CFU-Playground Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "proj_menu.h"

#include <stdio.h>

#include "cfu.h"
#include "menu.h"
#include "perf.h"
namespace {

  // Template Fn
static void time_fib(uint32_t n) {
  uint32_t r;
  unsigned t0 = perf_get_mcycle();
  r = cfu_op3_hw(0, n, 0);
  unsigned hw = perf_get_mcycle() - t0;

  t0 = perf_get_mcycle();
  r = cfu_op3_sw(0, n, 0);
  unsigned sw = perf_get_mcycle() - t0;

  printf("n=%lu hw=%u sw=%u result=%lu\n", n, hw, sw, r);
}

void do_time_fib(void) {
  time_fib(1);
  time_fib(10);
  time_fib(40);
}

void do_test(void) {  printf("print test\n"); }

enum { GM = 8, GN = 8, GK = 8, GEMM_REPS = 20 };

static int8_t A[GM * GK] __attribute__((aligned(4)));
static int8_t Bt[GN * GK] __attribute__((aligned(4)));

static void fill_gemm(void) {
  for (int i = 0; i < GM * GK; ++i) {
    A[i] = (int8_t)((i * 3) % 7 - 3);
  }
  for (int i = 0; i < GN * GK; ++i) {
    Bt[i] = (int8_t)((i * 5) % 5 - 2);
  }
}

static inline uint32_t load4(const int8_t* p) {
  uint32_t v;
  __asm__ volatile("lw %0, 0(%1)" : "=r"(v) : "r"(p));
  return v;
}

__attribute__((noinline)) static int32_t gemm_scalar(void) {
  int32_t sum = 0;
  for (int i = 0; i < GM; ++i) {
    for (int j = 0; j < GN; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < GK; ++k) {
        acc += (int32_t)A[i * GK + k] * (int32_t)Bt[j * GK + k];
      }
      sum += acc;
    }
  }
  return sum;
}

__attribute__((noinline)) static int32_t gemm_cfu(void) {
  int32_t sum = 0;
  for (int i = 0; i < GM; ++i) {
    for (int j = 0; j < GN; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < GK; k += 4) {
        uint32_t a = load4(&A[i * GK + k]);
        uint32_t b = load4(&Bt[j * GK + k]);
        acc += (int32_t)cfu_op4_hw(0, a, b);
      }
      sum += acc;
    }
  }
  return sum;
}

void do_gemm(void) {
  fill_gemm();
  static volatile int32_t sink;
  int32_t scalar_sum = 0;
  int32_t cfu_sum = 0;

  unsigned t0 = perf_get_mcycle();
  for (int r = 0; r < GEMM_REPS; ++r) {
    sink = gemm_scalar();
    scalar_sum += sink;
  }
  unsigned scalar_cycles = perf_get_mcycle() - t0;

  t0 = perf_get_mcycle();
  for (int r = 0; r < GEMM_REPS; ++r) {
    sink = gemm_cfu();
    cfu_sum += sink;
  }
  unsigned cfu_cycles = perf_get_mcycle() - t0;

  printf("gemm %dx%dx%d x%d  scalar=%u cfu=%u sum=%ld\n", GM, GN, GK,
         GEMM_REPS, scalar_cycles, cfu_cycles, (long)cfu_sum);
  if (scalar_sum != cfu_sum) {
    printf("mismatch scalar_sum=%ld cfu_sum=%ld\n", (long)scalar_sum,
           (long)cfu_sum);
  }
}

void do_hello_world(void) { puts("Hello, World!!!\n"); }

// Print fibonacci numbers
void do_print_fib(void) {
  puts("Fibonnaci Numbers");
  puts("-----------------");
  for (int i = 0; i < 8; i++) {
    printf("%2d:", i * 6);
    for (int j = 0; j < 6; j++) {
      printf(" %11lu", cfu_op3(0, i * 6 + j, 0));
    }
    puts("");
  }
}

struct Menu MENU = {
    "Project Menu",
    "project",
    {
        MENU_ITEM('0', "print fibonacci numbers", do_print_fib),
        MENU_ITEM('t', "time fib hw vs sw", do_time_fib),
        MENU_ITEM('p', "print test", do_test),
        MENU_ITEM('g', "gemm scalar vs cfu", do_gemm),
        MENU_ITEM('h', "say Hello", do_hello_world),
        MENU_END,
    },
};
};  // anonymous namespace

extern "C" void do_proj_menu() { menu_run(&MENU); }