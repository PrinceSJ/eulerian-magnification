import numpy as np
import scipy


def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    #print("")
    result = img * 1. / 255
    return result


def float_to_uint8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = img * 255
    return result


def float_to_int8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = (img * 255) - 127
    return result


def temporal_bandpass_filter(data, fps, freq_min, freq_max, amplification_factor,axis=0):
    print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis) #对实数序列Fourier变换
    print(data.shape)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    #print("Freq: "+str(frequencies))
    #print("Freq shape: "+str(frequencies.shape))
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    print("\nBoundLow= "+str(bound_low)+"\nBoundHigh= "+str(bound_high))
    print("fft dimension: "+str(fft.shape))
    #print("fft: "+str(fft[50,10,10,:]))
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    
    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    #print("result: "+str(result))
    result *= amplification_factor
    #print("New result: "+str(result))
    return result

#def TBP_Filter(vid, ):
