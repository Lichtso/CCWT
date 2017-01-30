#include <ccwt.h>
#include <png.h>

struct ccwt_render_png_data {
    unsigned char mode;
    png_bytepp row_pointers;
    png_structp png;
    png_infop png_info;
};

const unsigned int max_color_factor = 255;

#define ccwt_render_png_pixel_case(a, b, c) \
    pixel[0] = a*max_color_factor; \
    pixel[1] = b*max_color_factor; \
    pixel[2] = c*max_color_factor; \
    break

void ccwt_render_png_pixel(unsigned char* pixel, double H, double S, double V) {
    unsigned char h = H*6;
    double f = H*6-h, p = V*(1-S), q = V*(1-S*f), t = V*(1-(S*(1-f)));
    switch(h) {
        default: ccwt_render_png_pixel_case(V, t, p);
        case 1: ccwt_render_png_pixel_case(q, V, p);
        case 2: ccwt_render_png_pixel_case(p, V, t);
        case 3: ccwt_render_png_pixel_case(p, q, V);
        case 4: ccwt_render_png_pixel_case(t, p, V);
        case 5: ccwt_render_png_pixel_case(V, p, q);
    }
}

#define ccwt_render_png_row_case(instruction) \
    for(unsigned long x = 0; x < ccwt->output_width; ++x) \
        instruction; \
    break

#define clamp_and_scale(value) fmin(fmax(0.0, value), 1.0)*max_color_factor

int ccwt_render_png_row(struct ccwt_thread_data* thread, unsigned int y) {
    struct ccwt_data* ccwt = thread->ccwt;
    struct ccwt_render_png_data* render = (struct ccwt_render_png_data*)ccwt->user_data;
    unsigned char* output = render->row_pointers[y];
    switch(render->mode) {
        case REAL_GRAYSCALE:
            ccwt_render_png_row_case(output[x] = clamp_and_scale(0.5+0.5*creal(thread->output[ccwt->output_padding+x])));
        case IMAGINARY_GRAYSCALE:
            ccwt_render_png_row_case(output[x] = clamp_and_scale(0.5+0.5*cimag(thread->output[ccwt->output_padding+x])));
        case AMPLITUDE_GRAYSCALE:
            ccwt_render_png_row_case(output[x] = clamp_and_scale(cabs(thread->output[ccwt->output_padding+x])));
        case PHASE_GRAYSCALE:
            ccwt_render_png_row_case(output[x] = fabs(carg(thread->output[ccwt->output_padding+x])/M_PI)*max_color_factor);
        case EQUIPOTENTIAL:
            ccwt_render_png_row_case(ccwt_render_png_pixel(&output[x*3], fmin(cabs(thread->output[ccwt->output_padding+x])*0.9, 0.9), 1.0, 1.0));
        case RAINBOW_WALLPAPER:
            ccwt_render_png_row_case(ccwt_render_png_pixel(&output[x*3],
                carg(thread->output[ccwt->output_padding+x])/(2*M_PI)+0.5, 1.0,
                fmin(cabs(thread->output[ccwt->output_padding+x]), 1.0))
            );
    }
    return 0;
}

int ccwt_render_png(struct ccwt_data* ccwt, FILE* file, unsigned char mode) {
    int return_value = -1;
    struct ccwt_render_png_data render;
    render.mode = mode;
    render.row_pointers = (png_bytepp)malloc(ccwt->height*sizeof(void*));
    if(!render.row_pointers)
        return return_value;
    unsigned int bytesPerPixel = ((render.mode < 4) ? 1 : 3);
    for(unsigned int y = 0; y < ccwt->height; ++y) {
        render.row_pointers[y] = (png_bytep)malloc(ccwt->output_width*bytesPerPixel);
        if(!render.row_pointers[y])
            return return_value;
    }
    render.png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    render.png_info = png_create_info_struct(render.png);
    if(setjmp(png_jmpbuf(render.png))) {
        return_value = -3;
        goto cleanup;
    }
    png_init_io(render.png, file);
    png_set_IHDR(render.png, render.png_info, ccwt->output_width, ccwt->height,
                 8, (render.mode < 4) ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(render.png, render.png_info);
    ccwt->user_data = &render;
    ccwt->callback = ccwt_render_png_row;
    return_value = ccwt_calculate(ccwt);
    png_write_image(render.png, render.row_pointers);
    png_write_end(render.png, NULL);
    cleanup:
    for(unsigned int y = 0; y < ccwt->height; ++y)
        free(render.row_pointers[y]);
    free(render.row_pointers);
    png_destroy_write_struct(&render.png, &render.png_info);
    return return_value;
}
