#include <iostream>
#include <cstdlib>
// #include <cuda_fp16.h>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8
#define num_threads 256

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val[Z][3];
    if (tid >= width) return;

    __shared__ char mask[Z][Y][X];
    __shared__ unsigned char share_R[5][num_threads + 4];
    __shared__ unsigned char share_G[5][num_threads + 4];
    __shared__ unsigned char share_B[5][num_threads + 4];

    if (threadIdx.x == 0) {
        mask[0][0][0] = -1, mask[0][0][1] = -4, mask[0][0][2] = -6, mask[0][0][3] = -4, mask[0][0][4] = -1;
        mask[0][1][0] = -2, mask[0][1][1] = -8, mask[0][1][2] = -12, mask[0][1][3] = -8, mask[0][1][4] = -2;
        mask[0][2][0] = 0, mask[0][2][1] = 0, mask[0][2][2] = 0, mask[0][2][3] = 0, mask[0][2][4] = 0;
        mask[0][3][0] = 2, mask[0][3][1] = 8, mask[0][3][2] = 12, mask[0][3][3] = 8, mask[0][3][4] = 2;
        mask[0][4][0] = 1, mask[0][4][1] = 4, mask[0][4][2] = 6, mask[0][4][3] = 4, mask[0][4][4] = 1;
        mask[1][0][0] = -1, mask[1][0][1] = -2, mask[1][0][2] = 0, mask[1][0][3] = 2, mask[1][0][4] = 1;
        mask[1][1][0] = -4, mask[1][1][1] = -8, mask[1][1][2] = 0, mask[1][1][3] = 8, mask[1][1][4] = 4;
        mask[1][2][0] = -6, mask[1][2][1] = -12, mask[1][2][2] = 0, mask[1][2][3] = 12, mask[1][2][4] = 6;
        mask[1][3][0] = -4, mask[1][3][1] = -8, mask[1][3][2] = 0, mask[1][3][3] = 8, mask[1][3][4] = 4;
        mask[1][4][0] = -1, mask[1][4][1] = -2, mask[1][4][2] = 0, mask[1][4][3] = 2, mask[1][4][4] = 1;
    }

    const int offset = 2;
    for (int r = 0; r < 2; ++r) {
        share_R[r][threadIdx.x + offset] = 0;
        share_G[r][threadIdx.x + offset] = 0;
        share_B[r][threadIdx.x + offset] = 0;
    }
    if (threadIdx.x == 0) {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 2; ++c) {
                share_R[r][c] = 0;
                share_G[r][c] = 0;
                share_B[r][c] = 0;
            }
        }
        for (int r = 2; r < 5; ++r) {
            for (int c = 0; c < 2; ++c) {
                if (bound_check(tid + c - 2, 0, width)) {
                    share_R[r][c] = s[channels * (width * (r - 2) + (tid + c - 2)) + 2];
                    share_G[r][c] = s[channels * (width * (r - 2) + (tid + c - 2)) + 1];
                    share_B[r][c] = s[channels * (width * (r - 2) + (tid + c - 2)) + 0];
                } else {
                    share_R[r][c] = 0;
                    share_G[r][c] = 0;
                    share_B[r][c] = 0;
                }
            }
        }
    }
    if (threadIdx.x == num_threads - 1) {
        for (int r = 0; r < 2; ++r) {
            for (int c = num_threads + 2; c < num_threads + 4; ++c) {
                share_R[r][c] = 0;
                share_G[r][c] = 0;
                share_B[r][c] = 0;
            }
        }
        for (int r = 2; r < 5; ++r) {
            for (int c = num_threads + 2; c < num_threads + 4; ++c) {
                if (bound_check(tid + c - num_threads - 1, 0, width)) {
                    share_R[r][c] = s[channels * (width * (r - 2) + (tid + c - num_threads - 1)) + 2];
                    share_G[r][c] = s[channels * (width * (r - 2) + (tid + c - num_threads - 1)) + 1];
                    share_B[r][c] = s[channels * (width * (r - 2) + (tid + c - num_threads - 1)) + 0];
                } else {
                    share_R[r][c] = 0;
                    share_G[r][c] = 0;
                    share_B[r][c] = 0;
                }
            }
        }
    }
    for (int r = 0; r < 3; ++r) {
        share_R[r + offset][threadIdx.x + offset] = s[channels * (width * r + tid) + 2];
        share_G[r + offset][threadIdx.x + offset] = s[channels * (width * r + tid) + 1];
        share_B[r + offset][threadIdx.x + offset] = s[channels * (width * r + tid) + 0];
    }

    __syncthreads();

    int x = tid;
    for (int y = 0; y < height; ++y) {
        /* Z axis of mask */

        val[0][2] = val[1][2] = 0.;
        val[0][1] = val[1][1] = 0.;
        val[0][0] = val[1][0] = 0.;

        /* Y and X axis of mask */
        for (int v = -yBound; v <= yBound; ++v) {
            for (int u = -xBound; u <= xBound; ++u) {
                if (bound_check(x + u, 0, width) && bound_check(y + v, 0, height)) {
                    const int o = (y + v + offset) % 5;
                    const unsigned char R = share_R[o][threadIdx.x + u + offset];
                    const unsigned char G = share_G[o][threadIdx.x + u + offset];
                    const unsigned char B = share_B[o][threadIdx.x + u + offset];
                    val[0][2] += R * mask[0][u + xBound][v + yBound];
                    val[0][1] += G * mask[0][u + xBound][v + yBound];
                    val[0][0] += B * mask[0][u + xBound][v + yBound];

                    val[1][2] += R * mask[1][u + xBound][v + yBound];
                    val[1][1] += G * mask[1][u + xBound][v + yBound];
                    val[1][0] += B * mask[1][u + xBound][v + yBound];
                }
            }
        }

        float totalR = 0.;
        float totalG = 0.;
        float totalB = 0.;
        for (int i = 0; i < Z; ++i) {
            totalR += val[i][2] * val[i][2];
            totalG += val[i][1] * val[i][1];
            totalB += val[i][0] * val[i][0];
        }

        totalR = sqrt(totalR) / SCALE;
        totalG = sqrt(totalG) / SCALE;
        totalB = sqrt(totalB) / SCALE;

        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;

        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;

        if (bound_check(y + 3, 0, height)) {
            const int o = (y + 3 + offset) % 5;
            share_R[o][threadIdx.x + offset] = s[channels * (width * (y + 3) + tid) + 2];
            share_G[o][threadIdx.x + offset] = s[channels * (width * (y + 3) + tid) + 1];
            share_B[o][threadIdx.x + offset] = s[channels * (width * (y + 3) + tid) + 0];

            if (threadIdx.x == 0) {
                for (int c = 0; c < 2; ++c) {
                    if (bound_check(tid + c - 2, 0, width)) {
                        share_R[o][c] = s[channels * (width * (y + 3) + (tid + c - 2)) + 2];
                        share_G[o][c] = s[channels * (width * (y + 3) + (tid + c - 2)) + 1];
                        share_B[o][c] = s[channels * (width * (y + 3) + (tid + c - 2)) + 0];
                    } else {
                        share_R[o][c] = 0;
                        share_G[o][c] = 0;
                        share_B[o][c] = 0;
                    }
                }
            }
            if (threadIdx.x == num_threads - 1) {
                for (int c = num_threads + 2; c < num_threads + 4; ++c) {
                    if (bound_check(tid + c - num_threads - 1, 0, width)) {
                        share_R[o][c] = s[channels * (width * (y + 3) + (tid + c - num_threads - 1)) + 2];
                        share_G[o][c] = s[channels * (width * (y + 3) + (tid + c - num_threads - 1)) + 1];
                        share_B[o][c] = s[channels * (width * (y + 3) + (tid + c - num_threads - 1)) + 0];
                    } else {
                        share_R[o][c] = 0;
                        share_G[o][c] = 0;
                        share_B[o][c] = 0;
                    }
                }
            }
        }
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *) malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // decide to use how many blocks and threads
    const int num_blocks = width / num_threads + 1;

    // launch cuda kernel
    sobel <<<num_blocks, num_threads>>> (dsrc, ddst, height, width, channels);

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}

