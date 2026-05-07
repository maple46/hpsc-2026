#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    // Vectorize by AVX intrinsics
    __m512 xvec = _mm512_load_ps(x);
    __m512 yvec = _mm512_load_ps(y);
    __m512 mvec = _mm512_load_ps(m);

    // Create mask to exclude self-interaction (i != j)
    __m512i jvec = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512i ivec = _mm512_setr_epi32(i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i);
    __mmask16 mask = _mm512_cmpneq_epi32_mask(ivec, jvec);

    __m512 rxvec = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec); // float rx = x[i] - x[j];
    __m512 ryvec = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec); // float ry = y[i] - y[j];
    __m512 rvec_inv = _mm512_maskz_rsqrt14_ps(mask, _mm512_add_ps(_mm512_mul_ps(rxvec, rxvec), _mm512_mul_ps(ryvec, ryvec)));
    __m512 rvec_inv_cubed = _mm512_mul_ps(rvec_inv, rvec_inv);
    rvec_inv_cubed = _mm512_mul_ps(rvec_inv_cubed, rvec_inv); // 1 / (r * r * r)
    __m512 fxvec_tmp = _mm512_mul_ps(rxvec, rvec_inv_cubed); // tmp = rx / (r * r * r)
    __m512 fyvec_tmp = _mm512_mul_ps(ryvec, rvec_inv_cubed); // tmp = ry / (r * r * r)
    fxvec_tmp = _mm512_mul_ps(mvec, fxvec_tmp); // tmp = m * rx / (r * r * r)
    fyvec_tmp = _mm512_mul_ps(mvec, fyvec_tmp); // tmp = m * ry / (r * r * r)

    fx[i] -= _mm512_reduce_add_ps(fxvec_tmp);
    fy[i] -= _mm512_reduce_add_ps(fyvec_tmp);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
