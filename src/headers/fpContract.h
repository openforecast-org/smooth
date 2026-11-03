// No contraction of a*b+c into one fused multiply-add. A compiler that targets FMA
// (gcc on x86-64-v3, clang on arm64) fuses where its optimiser happens to see the
// pattern, so the R and the Python builds of the same C++, at their own optimisation
// levels, rounded differently. Included first in every translation unit of both
// packages, so that the code of Armadillo and Rcpp compiled there is covered too.
// Each compiler honours its own pragma (gcc ignores the standard one, clang gcc's),
// and MSVC does not contract under its default /fp:precise.
#ifndef SMOOTH_FP_CONTRACT_H
#define SMOOTH_FP_CONTRACT_H
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#endif
