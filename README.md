# CCWT library for C and Python
Complex [continuous wavelet transformation](https://en.wikipedia.org/wiki/Continuous_wavelet_transform)
- with a [gabor wavelet](https://en.wikipedia.org/wiki/Gabor_wavelet)
- interfaces for C99, python2.7 and python3.5
- using [libFFTW](http://www.fftw.org) for performance
- and [libPNG](http://www.libpng.org/pub/png/libpng.html) as possible output
- customizable frequency bands as well as linear and exponential helper methods


## Example

```python
import ccwt, numpy, math

height = 512
width = 512
border = 64

def generateWave(frequency_range, frequency_offset):
    phases = numpy.zeros(width)
    for t in range(0, width):
        phases[t] = (frequency_range*t/width+frequency_offset)*(math.pi*2.0*t/width)
    wave = numpy.cos(phases)
    for t in range(0, width):
        wave[t] *= (t > border) and (t < width-border)
    return wave

frequency_range = height*0.25
frequencies = ccwt.generate_frequencies(height, frequency_range)
wave = generateWave(frequency_range*0.5, 0.0)+generateWave(0.0, frequency_range*0.09375)+generateWave(0.0, frequency_range*(1.0-0.09375))

for mode in range(0, 6):
    ccwt.render_png('gallery/rendering_mode_'+str(mode)+'.png', mode, wave, frequencies)
```


## Documentation

### ccwt.calculate()
- input_array: Numpy 1D float64 array containing the signal
- frequency_range: Difference between the highest and the lowest frequency to analyze
- frequency_offset: Lowest frequency to analyze
- frequency_basis: Values > 0 switch from a linear to an exponential frequency scale using this as basis
- deviation: Values near 0 have better frequency resolution, values towards infinity have better time resolution
- padding: Zero samples to be virtually added at each end of the input signal
- width: Width of the resulting image in pixels, this can be the input sample count or lower for downsampling
- height: Height of the resulting image in pixels, this is also the number of frequencies to analyze

### ccwt.render_png()
Same as calculate but with these two at the end:
- rendering_mode: 0 to 5 indicating the color scheme for rendering
- path: Filename of the resulting PNG image
