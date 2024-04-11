#include <ccwt.h>
#include <png.h>

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

double ccwt_clamp(unsigned int x, unsigned int maxX, double deltaX) {
    return fmin(fmax(0.0, ((double)x)-((double)maxX)*deltaX), (double)(maxX-1));
}

complex_type cctw_interpolate_linear(complex_type low, complex_type high, double x) {
    return low+(high-low)*x;
}

double ccwt_render_png_remap_output_linear(double log_factor, double value) {
    double sign = (0 < value)-(value < 0);
    return fmin(value*sign, 1.0)*sign;
}

double ccwt_render_png_remap_output_logarithmic(double log_factor, double value) {
    double sign = (0 < value)-(value < 0);
    value = log(value*sign)*log_factor;
    return fmin(fmax(0.0, value), 1.0)*sign;
}

complex_type ccwt_synchrosqueeze_identity(struct ccwt_data* ccwt, unsigned int x, unsigned int y) {
    complex_type* spectrogram = (complex_type*)ccwt->user_data;
    return spectrogram[ccwt->output_width*y+x];
}

complex_type ccwt_synchrosqueeze_derivative_quotient(struct ccwt_data* ccwt, unsigned int x, unsigned int y) {
    complex_type* spectrogram = (complex_type*)ccwt->user_data;
    complex_type next = (y == 0) ? spectrogram[ccwt->output_width+x] : spectrogram[ccwt->output_width*(y-1)+x];
    return (1 - next / spectrogram[ccwt->output_width*y+x]) * M_PI * (y == 0 ? -2.0 : 2.0);
}

complex_type ccwt_synchrosqueeze_nearest_sample(struct ccwt_data* ccwt, unsigned int x, unsigned int y) {
    double strength = cabs(ccwt_synchrosqueeze_identity(ccwt, x, y));
    complex_type derivative_quotient = ccwt_synchrosqueeze_derivative_quotient(ccwt, x, y);
    x = (unsigned int)ccwt_clamp(x, ccwt->output_width, cimag(derivative_quotient) * strength);
    y = (unsigned int)ccwt_clamp(y, ccwt->output_width, creal(derivative_quotient) * strength);
    return ccwt_synchrosqueeze_identity(ccwt, x, y);
}

complex_type ccwt_synchrosqueeze_linear_sample(struct ccwt_data* ccwt, unsigned int x, unsigned int y) {
    double strength = cabs(ccwt_synchrosqueeze_identity(ccwt, x, y));
    complex_type derivative_quotient = ccwt_synchrosqueeze_derivative_quotient(ccwt, x, y);
    double intX = ccwt_clamp(x, ccwt->output_width, cimag(derivative_quotient) * strength);
    double intY = ccwt_clamp(y, ccwt->output_width, creal(derivative_quotient) * strength);
    double fracX = modf(intX, &intX);
    double fracY = modf(intY, &intY);
    x = (unsigned int)intX;
    y = (unsigned int)intY;
    return cctw_interpolate_linear(
        cctw_interpolate_linear(
            ccwt_synchrosqueeze_identity(ccwt, x, y),
            ccwt_synchrosqueeze_identity(ccwt, x+1, y),
            fracX
        ),
        cctw_interpolate_linear(
            ccwt_synchrosqueeze_identity(ccwt, x, y+1),
            ccwt_synchrosqueeze_identity(ccwt, x+1, y+1),
            fracX
        ),
        fracY
    );
}

int ccwt_copy_output_row(struct ccwt_thread_data* thread, unsigned int y) {
    struct ccwt_data* ccwt = thread->ccwt;
    complex_type* spectrogram = (complex_type*)ccwt->user_data;
    for(unsigned long x = 0; x < ccwt->output_width; ++x)
        spectrogram[ccwt->output_width*y+x] = thread->output[ccwt->output_padding+x];
    return 0;
}

int ccwt_render_png(struct ccwt_data* ccwt, FILE* file, unsigned char synchrosqueeze_mode, unsigned char color_scheme, double logarithmic_basis) {
    int return_value = -1;
    complex_type (*synchrosqueeze)(struct ccwt_data* ccwt, unsigned int, unsigned int) = ccwt_synchrosqueeze_identity;
    switch (synchrosqueeze_mode) {
        case 1:
            synchrosqueeze = ccwt_synchrosqueeze_derivative_quotient;
            break;
        case 2:
            synchrosqueeze = ccwt_synchrosqueeze_nearest_sample;
            break;
        case 3:
            synchrosqueeze = ccwt_synchrosqueeze_linear_sample;
            break;
    }
    double log_factor = 0.0;
    double (*remap_output)(double, double) = ccwt_render_png_remap_output_linear;
    if(logarithmic_basis > 0.0) {
        log_factor = 1.0/log(logarithmic_basis);
        remap_output = ccwt_render_png_remap_output_logarithmic;
    }
    complex_type* spectrogram = (complex_type*)malloc(ccwt->output_width*ccwt->height*sizeof(complex_type));
    png_bytepp row_pointers = (png_bytepp)malloc(ccwt->height*sizeof(void*));
    if(!row_pointers)
        return return_value;
    unsigned int bytesPerPixel = ((color_scheme < 4) ? 1 : 3);
    for(unsigned int y = 0; y < ccwt->height; ++y) {
        row_pointers[y] = (png_bytep)malloc(ccwt->output_width*bytesPerPixel);
        if(!row_pointers[y])
            return return_value;
    }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop png_info = png_create_info_struct(png);
    if(setjmp(png_jmpbuf(png))) {
        return_value = -3;
        goto cleanup;
    }
    png_init_io(png, file);
    png_set_IHDR(png, png_info, ccwt->output_width, ccwt->height,
                 8, (color_scheme < 4) ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png, png_info);
    ccwt->user_data = spectrogram;
    ccwt->callback = ccwt_copy_output_row;
    return_value = ccwt_numeric_output(ccwt);
    for(unsigned int y = 0; y < ccwt->height; ++y) {
        unsigned char* output = row_pointers[y];
        switch(color_scheme) {
            case REAL_GRAYSCALE:
                ccwt_render_pixel_grayscale(0.5+0.5*remap_output(log_factor, creal(synchrosqueeze(ccwt, x, y))));
            case IMAGINARY_GRAYSCALE:
                ccwt_render_pixel_grayscale(0.5+0.5*remap_output(log_factor, cimag(synchrosqueeze(ccwt, x, y))));
            case AMPLITUDE_GRAYSCALE:
                ccwt_render_pixel_grayscale(remap_output(log_factor, cabs(synchrosqueeze(ccwt, x, y))));
            case PHASE_GRAYSCALE:
                ccwt_render_pixel_grayscale(fabs(carg(synchrosqueeze(ccwt, x, y))/M_PI));
            case EQUIPOTENTIAL:
                ccwt_render_pixel_color(
                    remap_output(log_factor, cabs(synchrosqueeze(ccwt, x, y)))*0.9,
                    1.0, 1.0
                );
            case RAINBOW_WALLPAPER:
                ccwt_render_pixel_color(
                    carg(synchrosqueeze(ccwt, x, y))/(2*M_PI)+0.5, 1.0,
                    remap_output(log_factor, cabs(synchrosqueeze(ccwt, x, y)))
                );
        }
    }
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);
    cleanup:
    for(unsigned int y = 0; y < ccwt->height; ++y)
        free(row_pointers[y]);
    free(row_pointers);
    free(spectrogram);
    png_destroy_write_struct(&png, &png_info);
    return return_value;
}
