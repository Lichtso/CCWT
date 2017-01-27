#include <ccwt.h>
#include <png.h>

struct ccwt_render_png_data {
    unsigned char mode;
    unsigned char* row;
    png_structp png;
    png_infop png_info;
};

const unsigned int max_color_factor = 255;

#define ccwt_render_png_pixel_case(a, b, c) \
    pixel[0] = a*max_color_factor; \
    pixel[1] = b*max_color_factor; \
    pixel[2] = c*max_color_factor; \
    break;

void ccwt_render_png_pixel(unsigned char* pixel, double H, double S, double V) {
    unsigned char h = H*6;
    double f = H*6-h, p = V*(1-S), q = V*(1-S*f), t = V*(1-(S*(1-f)));
    switch(h) {
        default: ccwt_render_png_pixel_case(V, t, p)
        case 1: ccwt_render_png_pixel_case(q, V, p)
        case 2: ccwt_render_png_pixel_case(p, V, t)
        case 3: ccwt_render_png_pixel_case(p, q, V)
        case 4: ccwt_render_png_pixel_case(t, p, V)
        case 5: ccwt_render_png_pixel_case(V, p, q)
    }
}

#define ccwt_render_png_row_case(instruction) \
    for(unsigned long x = 0; x < ccwt->output_width; ++x) \
        instruction; \
    break

#define clamp_and_scale(value) fmin(fmax(0.0, value), 1.0)*max_color_factor

int ccwt_render_png_row(struct ccwt_data* ccwt, void* user_data, unsigned int row) {
    struct ccwt_render_png_data* render = (struct ccwt_render_png_data*)user_data;
    switch(render->mode) {
        case 0: // Real Grayscale
            ccwt_render_png_row_case(render->row[x] = clamp_and_scale(0.5+0.5*creal(ccwt->output[ccwt->output_padding+x])));
        case 1: // Imaginary Grayscale
            ccwt_render_png_row_case(render->row[x] = clamp_and_scale(0.5+0.5*cimag(ccwt->output[ccwt->output_padding+x])));
        case 2: // Amplitude Grayscale
            ccwt_render_png_row_case(render->row[x] = clamp_and_scale(cabs(ccwt->output[ccwt->output_padding+x])));
        case 3: // Phase Grayscale
            ccwt_render_png_row_case(render->row[x] = fabs(carg(ccwt->output[ccwt->output_padding+x])/M_PI)*max_color_factor);
        case 4: // Equipotential
            ccwt_render_png_row_case(ccwt_render_png_pixel(&render->row[x*3], fmin(cabs(ccwt->output[ccwt->output_padding+x])*0.9, 0.9), 1.0, 1.0));
        case 5: // Rainbow Wallpaper
            ccwt_render_png_row_case(ccwt_render_png_pixel(&render->row[x*3],
                carg(ccwt->output[ccwt->output_padding+x])/(2*M_PI)+0.5, 1.0,
                fmin(cabs(ccwt->output[ccwt->output_padding+x]), 1.0))
            );
    }
    png_write_row(render->png, render->row);
    return 0;
}

int ccwt_render_png(struct ccwt_data* ccwt, FILE* file, unsigned char mode) {
    struct ccwt_render_png_data render;
    render.mode = mode;
    render.row = (unsigned char*)malloc(ccwt->output_width*((render.mode < 4) ? 1 : 3));
    if(!render.row)
        return -1;
    render.png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    render.png_info = png_create_info_struct(render.png);
    if(setjmp(png_jmpbuf(render.png))) {
        free(render.row);
        png_destroy_write_struct(&render.png, &render.png_info);
        return -2;
    }
    png_init_io(render.png, file);
    png_set_IHDR(render.png, render.png_info, ccwt->output_width, ccwt->height,
                 8, (render.mode < 4) ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(render.png, render.png_info);
    int return_value = ccwt_calculate(ccwt, &render, ccwt_render_png_row);
    png_write_end(render.png, NULL);
    free(render.row);
    png_destroy_write_struct(&render.png, &render.png_info);
    return return_value;
}
