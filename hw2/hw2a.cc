#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>

int curr_height = 0, curr_width = 0;
int iters;
int width, height;
int* image;
double lower, upper, left, right, x_offset, y_offset;
pthread_mutex_t mutex;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    // assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    // assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    // assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* calc(void* arg) {
    while (true) {
        int w1, h1, w2, h2;
        pthread_mutex_lock(&mutex);
        if (curr_width == width) {
            curr_width = 0;
            curr_height++;
        }
        w1 = curr_width++, h1 = curr_height;
        if (curr_width == width) {
            curr_width = 0;
            curr_height++;
        }
        w2 = curr_width++, h2 = curr_height;
        pthread_mutex_unlock(&mutex);

        if (h1 == height) {
            break;
        }
        double _x1 = left + w1 * x_offset;
        double _y1 = lower + h1 * y_offset;
        double _x2 = left + w2 * x_offset;
        double _y2 = lower + h2 * y_offset;

        if (h2 == height) {
            double x = 0, y = 0;
            double length_squared = 0;
            int count = 0;
            while (count < iters && length_squared < 4.0) {
                double tmp = x * x - y * y + _x1;
                y = 2 * x * y + _y1;
                x = tmp;
                length_squared = x * x + y * y;
                count++;
            }
            image[h1 * width + w1] = count;
            continue;
        }
        
        __m128d x0 = _mm_set_pd(_x1, _x2);
        __m128d y0 = _mm_set_pd(_y1, _y2);
        __m128d x = _mm_set_sd(0);
        __m128d y = _mm_set_sd(0);
        __m128d length_squared = _mm_set_sd(0);
        int count[2] = {0, 0};
        int check[2] = {0, 0};

        while (!check[0] || !check[1]) {
            __m128d comp_res = _mm_cmplt_pd(length_squared, _mm_set_pd1(4));
            if (!check[1]) {
                if (count[1] < iters && _mm_cvtsd_f64(comp_res) != 0) {
                    count[1]++;
                } else {
                    check[1] = 1;
                }
            }
            if (!check[0]) {
                if (count[0] < iters && _mm_cvtsd_f64(_mm_unpackhi_pd(comp_res, comp_res)) != 0) {
                    count[0]++;
                } else {
                    check[0] = 1;
                }
            }
            __m128d tmp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), x0);
            y = _mm_add_pd(_mm_mul_pd(_mm_set_pd1(2), _mm_mul_pd(x, y)), y0);
            x = _mm_add_pd(tmp, _mm_set_sd(0));
            length_squared = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
        }

        image[h1 * width + w1] = count[0];
        image[h2 * width + w2] = count[1];
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    const int thread_num = CPU_COUNT(&cpu_set);

    /* argument parsing */
    // assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    x_offset = (right - left) / width;
    y_offset = (upper - lower) / height;
    pthread_mutex_init(&mutex, 0);

    /* allocate memory for image */
    image = (int*) malloc(width * height * sizeof(int));
    // assert(image);

    pthread_t threads[thread_num];

    for (int t = 0; t < thread_num; ++t) {
        pthread_create(&threads[t], NULL, calc, NULL);
    }

    for (int t = 0; t < thread_num; ++t) {
        pthread_join(threads[t], NULL);
    }
    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    pthread_exit(NULL);
}
