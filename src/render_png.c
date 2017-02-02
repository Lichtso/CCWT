#include <ccwt.h>
#include <png.h>

struct ccwt_render_png_data {
    unsigned char mode;
    double log_factor;
    double (*remap)(struct ccwt_render_png_data*, double);
    png_bytepp row_pointers;
    png_structp png;
    png_infop png_info;
};

const unsigned int max_color_factor = 255;

#define ccwt_render_png_pixel_case(r, g, b) \
    pixel[0] = r; pixel[1] = g; pixel[2] = b; \
    break

void ccwt_render_png_HSV2RGB(unsigned char* pixel, double H, double S, double V) {
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

#define ccwt_render_pixel_grayscale(value) \
    ccwt_render_png_row_case(output[x] = (value)*max_color_factor);

#define ccwt_render_pixel_color(H, S, V) \
    ccwt_render_png_row_case(ccwt_render_png_HSV2RGB(&output[x*3], H, S, V*max_color_factor));

double ccwt_render_png_remap_linear(struct ccwt_render_png_data* render, double value) {
    double sign = (0 < value)-(value < 0);
    return fmin(value*sign, 1.0)*sign;
}

double ccwt_render_png_remap_logarithmic(struct ccwt_render_png_data* render, double value) {
    double sign = (0 < value)-(value < 0);
    value = log(value*sign)*render->log_factor;
    return fmin(fmax(0.0, value), 1.0)*sign;
}

int ccwt_render_png_row(struct ccwt_thread_data* thread, unsigned int y) {
    struct ccwt_data* ccwt = thread->ccwt;
    struct ccwt_render_png_data* render = (struct ccwt_render_png_data*)ccwt->user_data;
    unsigned char* output = render->row_pointers[y];
    switch(render->mode) {
        case REAL_GRAYSCALE:
            ccwt_render_pixel_grayscale(0.5+0.5*render->remap(render, creal(thread->output[ccwt->output_padding+x])));
        case IMAGINARY_GRAYSCALE:
            ccwt_render_pixel_grayscale(0.5+0.5*render->remap(render, cimag(thread->output[ccwt->output_padding+x])));
        case AMPLITUDE_GRAYSCALE:
            ccwt_render_pixel_grayscale(render->remap(render, cabs(thread->output[ccwt->output_padding+x])));
        case PHASE_GRAYSCALE:
            ccwt_render_pixel_grayscale(fabs(carg(thread->output[ccwt->output_padding+x])/M_PI));
        case EQUIPOTENTIAL:
            ccwt_render_pixel_color(
                render->remap(render, cabs(thread->output[ccwt->output_padding+x]))*0.9,
                1.0, 1.0
            );
        case RAINBOW_WALLPAPER:
            ccwt_render_pixel_color(
                carg(thread->output[ccwt->output_padding+x])/(2*M_PI)+0.5, 1.0,
                render->remap(render, cabs(thread->output[ccwt->output_padding+x]))
            );
    }
    return 0;
}

int ccwt_render_png(struct ccwt_data* ccwt, FILE* file, unsigned char mode, double logarithmic_basis) {
    int return_value = -1;
    struct ccwt_render_png_data render;
    render.mode = mode;
    if(logarithmic_basis <= 0.0) {
        render.log_factor = 0.0;
        render.remap = ccwt_render_png_remap_linear;
    } else {
        render.log_factor = 1.0/log(logarithmic_basis);
        render.remap = ccwt_render_png_remap_logarithmic;
    }
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
    return_value = ccwt_numeric_output(ccwt);
    png_write_image(render.png, render.row_pointers);
    png_write_end(render.png, NULL);
    cleanup:
    for(unsigned int y = 0; y < ccwt->height; ++y)
        free(render.row_pointers[y]);
    free(render.row_pointers);
    png_destroy_write_struct(&render.png, &render.png_info);
    return return_value;
}
