#include <immintrin.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <fstream>
#include <random>

static inline float _mm256_addv_ps(const __m256 v) {
    __m128 res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#define extract_low_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define extract_high_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define extract_low_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define extract_high_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))

template <int N>
struct SignedWideningAdder {
    __m256i lhs_low, lhs_high;
    inline void push(__m256i v, int k) {
        if (k == 0) { lhs_low = extract_low_epi8_epi16(v); lhs_high = extract_high_epi8_epi16(v); }
        else { lhs_low = _mm256_add_epi16(lhs_low, extract_low_epi8_epi16(v));
               lhs_high = _mm256_add_epi16(lhs_high, extract_high_epi8_epi16(v)); }
    }
    inline __m256i get_low() { return lhs_low; }
    inline __m256i get_high() { return lhs_high; }
};

inline void partial_max_g4_int8_k8(float* lut_scales, float* b) {
    const __m256i vec_bi = _mm256_set_epi32(112,96,80,64,48,32,16,0);
    __m256 vec_b0=_mm256_i32gather_ps(b+0,vec_bi,1), vec_b1=_mm256_i32gather_ps(b+1,vec_bi,1);
    __m256 vec_b2=_mm256_i32gather_ps(b+2,vec_bi,1), vec_b3=_mm256_i32gather_ps(b+3,vec_bi,1);
    const __m256 sign=_mm256_set1_ps(-0.0f);
    __m256 a0=_mm256_andnot_ps(sign,vec_b0), a1=_mm256_andnot_ps(sign,vec_b1);
    __m256 a2=_mm256_andnot_ps(sign,vec_b2), a3=_mm256_andnot_ps(sign,vec_b3);
    __m256 s=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
    __m128 m4=_mm_max_ps(_mm256_extractf128_ps(s,1),_mm256_castps256_ps128(s));
    m4=_mm_max_ps(m4,_mm_movehl_ps(m4,m4)); m4=_mm_max_ss(m4,_mm_movehdup_ps(m4));
    float sc=_mm_cvtss_f32(m4)/127.0f; *lut_scales=std::max(*lut_scales,sc);
}

inline void lut_ctor_g4_int8_avx2(int32_t act_k, int8_t* qlut, float* b, float* lut_scales, float* lut_biases) {
    __m256 vec_lut[16]; float biases=0.0f;
    const __m256i vec_bi=_mm256_set_epi32(112,96,80,64,48,32,16,0);
    float scales=*lut_scales, t_scales=scales?1.0f/scales:0.0f;
    for (int k=0;k<act_k/32;++k) {
        __m256 vb0=_mm256_i32gather_ps(b+k*32+0,vec_bi,1), vb1=_mm256_i32gather_ps(b+k*32+1,vec_bi,1);
        __m256 vb2=_mm256_i32gather_ps(b+k*32+2,vec_bi,1), vb3=_mm256_i32gather_ps(b+k*32+3,vec_bi,1);
        for (int g=1;g<16;g+=2) {
            vec_lut[g]=vb0;
            if(g&0b0010) vec_lut[g]=_mm256_add_ps(vec_lut[g],vb1); else vec_lut[g]=_mm256_sub_ps(vec_lut[g],vb1);
            if(g&0b0100) vec_lut[g]=_mm256_add_ps(vec_lut[g],vb2); else vec_lut[g]=_mm256_sub_ps(vec_lut[g],vb2);
            if(g&0b1000) vec_lut[g]=_mm256_add_ps(vec_lut[g],vb3); else vec_lut[g]=_mm256_sub_ps(vec_lut[g],vb3);
        }
        for (int g=0;g<16;g+=2) vec_lut[g]=_mm256_sub_ps(_mm256_setzero_ps(),vec_lut[15-g]);
        biases += _mm256_addv_ps(vec_lut[0]);
        for (int g=0;g<16;++g) vec_lut[g]=_mm256_mul_ps(vec_lut[g],_mm256_set1_ps(t_scales));
        __m256i vec_qlut[4];
        const __m256i shuf=_mm256_setr_epi8(0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15,0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15);
        for (int g=0;g<4;++g) {
            __m256i i0=_mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g*4+0],_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            __m256i i1=_mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g*4+1],_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            __m256i i2=_mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g*4+2],_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            __m256i i3=_mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g*4+3],_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
            i0=_mm256_packs_epi32(i0,i1); i2=_mm256_packs_epi32(i2,i3); i0=_mm256_packs_epi16(i0,i2);
            vec_qlut[g]=_mm256_shuffle_epi8(i0,shuf);
        }
        int32_t* qi=reinterpret_cast<int32_t*>(qlut);
        for (int g=0;g<4;++g) qi[k*32+0*4+g]=_mm256_extract_epi32(vec_qlut[g],0);
        for (int g=0;g<4;++g) qi[k*32+1*4+g]=_mm256_extract_epi32(vec_qlut[g],1);
        for (int g=0;g<4;++g) qi[k*32+2*4+g]=_mm256_extract_epi32(vec_qlut[g],2);
        for (int g=0;g<4;++g) qi[k*32+3*4+g]=_mm256_extract_epi32(vec_qlut[g],3);
        for (int g=0;g<4;++g) qi[k*32+4*4+g]=_mm256_extract_epi32(vec_qlut[g],4);
        for (int g=0;g<4;++g) qi[k*32+5*4+g]=_mm256_extract_epi32(vec_qlut[g],5);
        for (int g=0;g<4;++g) qi[k*32+6*4+g]=_mm256_extract_epi32(vec_qlut[g],6);
        for (int g=0;g<4;++g) qi[k*32+7*4+g]=_mm256_extract_epi32(vec_qlut[g],7);
    }
    *lut_scales=scales; *lut_biases=biases;
}

template <int Bits, int ActK=32>
inline void tbl_update_avx2(int32_t m,int32_t k_groups,float* c,int8_t* lut,uint8_t* a,
                            float* scales,float* lut_scales,float* lut_biases) {
    const __m128i vec_mask=_mm_set1_epi8(0x0f);
    __m128i* vec_lut=(__m128i*)_mm_malloc(k_groups*sizeof(__m128i),32);
    for (int k=0;k<k_groups;k++) vec_lut[k]=_mm_loadu_si128(reinterpret_cast<__m128i*>(lut+k*16));
    SignedWideningAdder<ActK> adder;
    for (int i=0;i<m/2;i+=16) {
        __m256 c0=_mm256_setzero_ps(),c1=_mm256_setzero_ps(),c2=_mm256_setzero_ps(),c3=_mm256_setzero_ps();
        for (int kk=0;kk<k_groups;kk+=ActK) {
            for (int k=0;k<ActK;k++) {
                __m128i as=_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+i*k_groups+(kk+k)*16));
                __m128i bot=_mm_and_si128(as,vec_mask), top=_mm_and_si128(_mm_srli_epi16(as,4),vec_mask);
                __m256i lut_=_mm256_set_m128i(vec_lut[kk+k],vec_lut[kk+k]);
                __m256i av=_mm256_set_m128i(top,bot);
                adder.push(_mm256_shuffle_epi8(lut_,av),k);
            }
            __m256 v00=_mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_low()));
            __m256 v01=_mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_low()));
            __m256 v10=_mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_high()));
            __m256 v11=_mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_high()));
            int gi=kk/ActK; float ls=lut_scales[gi], lb=lut_biases[gi];
            #define F(vs,ib) ((ib)%Bits) ? _mm256_mul_ps((vs),_mm256_set1_ps(ls)) : _mm256_fmadd_ps((vs),_mm256_set1_ps(ls),_mm256_set1_ps(lb))
            if (kk==0) { c0=F(v00,(i/4)); c1=F(v01,(i/4+1)); c2=F(v10,(i/4+2)); c3=F(v11,(i/4+3)); }
            else { c0=_mm256_add_ps(c0,F(v00,(i/4))); c1=_mm256_add_ps(c1,F(v01,(i/4+1)));
                   c2=_mm256_add_ps(c2,F(v10,(i/4+2))); c3=_mm256_add_ps(c3,F(v11,(i/4+3))); }
            #undef F
        }
        __m256 s0=_mm256_loadu_ps(scales+((i/4)/Bits)*8), s1=_mm256_loadu_ps(scales+((i/4+1)/Bits)*8);
        __m256 s2=_mm256_loadu_ps(scales+((i/4+2)/Bits)*8), s3=_mm256_loadu_ps(scales+((i/4+3)/Bits)*8);
        _mm256_storeu_ps(c+i*2,   _mm256_fmadd_ps(c0,s0,_mm256_loadu_ps(c+i*2)));
        _mm256_storeu_ps(c+i*2+8, _mm256_fmadd_ps(c1,s1,_mm256_loadu_ps(c+i*2+8)));
        _mm256_storeu_ps(c+i*2+16,_mm256_fmadd_ps(c2,s2,_mm256_loadu_ps(c+i*2+16)));
        _mm256_storeu_ps(c+i*2+24,_mm256_fmadd_ps(c3,s3,_mm256_loadu_ps(c+i*2+24)));
    }
    _mm_free(vec_lut);
}

// 正向打包：W_bit[m][k] ∈ {0,1} -> T-MAC 交錯格式
void pack_tmac_weights(int M,int K,const std::vector<uint8_t>& W_bit,uint8_t* W_packed) {
    int K_GROUPS=K/4;
    std::memset(W_packed,0,(M/32)*K_GROUPS*16);
    for (int m=0;m<M;++m) {
        int block_m=m/32, mwb=m%32, group=mwb/8, lane=mwb%8;
        int byte_in_block=(group==0||group==2)?lane:(8+lane);
        bool top_nibble=(group==2||group==3);
        for (int kt=0;kt<K_GROUPS;++kt) {
            uint8_t b0=W_bit[m*K+kt*4+0], b1=W_bit[m*K+kt*4+1];
            uint8_t b2=W_bit[m*K+kt*4+2], b3=W_bit[m*K+kt*4+3];
            uint8_t idx=(b3<<3)|(b2<<2)|(b1<<1)|b0;
            int off=block_m*(K_GROUPS*16)+kt*16+byte_in_block;
            if (top_nibble) W_packed[off]|=(idx<<4); else W_packed[off]|=(idx&0x0F);
        }
    }
}

int main() {
    const int Bits=4, ActK=32;
    std::vector<int> sizes={256,1024,2048,4096,8192};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> act_dist(-5.0f,5.0f);
    std::uniform_int_distribution<int> bit_dist(0,1);

    std::cout << "=== t_mac_input: 產生資料、打包、跑 AVX2 kernel ===\n";

    for (int size : sizes) {
        int m=size, k=size;
        int k_groups=k/4, num_groups=k_groups/ActK, group_span=ActK*4;

        std::vector<uint8_t> W_bit(m*k);
        for (auto& b : W_bit) b=(uint8_t)bit_dist(rng);
        std::vector<float> activations(k);
        for (auto& a : activations) a=act_dist(rng);

        uint8_t* weights=(uint8_t*)_mm_malloc(m*k_groups*16*sizeof(uint8_t),32);
        pack_tmac_weights(m,k,W_bit,weights);

        int8_t* qlut=(int8_t*)_mm_malloc(k_groups*ActK*16*sizeof(int8_t),32);
        float* out_c=(float*)_mm_malloc(m*sizeof(float),32);
        float* scales=(float*)_mm_malloc((m/Bits)*16*sizeof(float),32);
        float* lut_scales=(float*)_mm_malloc(num_groups*sizeof(float),32);
        float* lut_biases=(float*)_mm_malloc(num_groups*sizeof(float),32);
        std::memset(out_c,0,m*sizeof(float));
        for (int i=0;i<(m/Bits)*16;++i) scales[i]=1.0f;

        for (int g=0;g<num_groups;++g) {
            lut_scales[g]=0.0f;
            for (int sub=0;sub<group_span/32;++sub)
                partial_max_g4_int8_k8(&lut_scales[g],activations.data()+g*group_span+sub*32);
            lut_ctor_g4_int8_avx2(group_span,qlut+g*(ActK*16),activations.data()+g*group_span,&lut_scales[g],&lut_biases[g]);
        }

        int iterations=20;
        auto start=std::chrono::high_resolution_clock::now();
        for (int it=0;it<iterations;++it) {
            std::memset(out_c,0,m*sizeof(float));
            tbl_update_avx2<Bits,ActK>(m,k_groups,out_c,qlut,weights,scales,lut_scales,lut_biases);
        }
        auto end=std::chrono::high_resolution_clock::now();
        double avg_ms=std::chrono::duration<double,std::milli>(end-start).count()/iterations;
        double gflops=(2.0*m*k)/(avg_ms/1000.0)/1e9;
        std::cout << size << "x" << size << "  AVX2 latency=" << avg_ms << "ms  " << gflops << " GFLOPS\n";

        // 寫出：原始輸入(W_bit, activations) + AVX2輸出，供 naive_triple_loop.cpp 獨立讀取比對
        std::string tag=std::to_string(size);
        std::ofstream fw("data_W_bit_"+tag+".bin",std::ios::binary);
        fw.write((char*)W_bit.data(),W_bit.size());
        std::ofstream fa("data_act_"+tag+".bin",std::ios::binary);
        fa.write((char*)activations.data(),activations.size()*sizeof(float));
        std::ofstream fo("data_avx2_out_"+tag+".bin",std::ios::binary);
        fo.write((char*)out_c,m*sizeof(float));

        _mm_free(weights); _mm_free(qlut); _mm_free(out_c);
        _mm_free(scales); _mm_free(lut_scales); _mm_free(lut_biases);
    }
    return 0;
}