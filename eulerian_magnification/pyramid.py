import numpy
import cv2


def create_gaussian_image_pyramid(image, pyramid_levels):
    gauss_copy = numpy.ndarray(shape=image.shape, dtype="float")
    gauss_copy[:] = image
    img_pyramid = [gauss_copy]
    for pyramid_level in range(1, pyramid_levels):
        #print("origin: "+str(type(gauss_copy)))
        gauss_copy = cv2.pyrDown(gauss_copy)
        #print("after: "+str(gauss_copy.shape))
        img_pyramid.append(gauss_copy)

    return img_pyramid


def create_laplacian_image_pyramid(image, pyramid_levels):
    #print(image.shape,image.dtype)
    gauss_pyramid = create_gaussian_image_pyramid(image, pyramid_levels)
    laplacian_pyramid = []
    for i in range(pyramid_levels - 1):
        laplacian_pyramid.append((gauss_pyramid[i] - cv2.pyrUp(gauss_pyramid[i + 1])))

    laplacian_pyramid.append(gauss_pyramid[-1])
    return laplacian_pyramid


def create_gaussian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, create_gaussian_image_pyramid)


def create_laplacian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, create_laplacian_image_pyramid)


def _create_pyramid(video, pyramid_levels, pyramid_fn):
    vid_pyramid = []
    print(video.shape)
    # frame_count, height, width, colors = video.shape
    for frame_number, frame in enumerate(video):
        #print(frame_number, frame)
        frame_pyramid = pyramid_fn(frame, pyramid_levels)   #laplacian pyramid
        #print(sizeof(frame_pyramid))
        for pyramid_level, pyramid_sub_frame in enumerate(frame_pyramid):
            if frame_number == 0:
                #print(video.shape[0], pyramid_sub_frame.shape[0], pyramid_sub_frame.shape[1])
                vid_pyramid.append(numpy.zeros((video.shape[0], pyramid_sub_frame.shape[0], pyramid_sub_frame.shape[1], 3),
                    dtype="float16"))   #RAM too small for float64 

            vid_pyramid[pyramid_level][frame_number] = pyramid_sub_frame

    return vid_pyramid


def collapse_laplacian_pyramid(image_pyramid):
    img = image_pyramid.pop()
    # print("img: "+str(img.shape))
    # imgtemp = numpy.ndarray(shape=img.shape, dtype="float")
    # imgtemp = cv2.pyrUp(img.astype('float64'))
    # print("pyrUpimg: "+str(imgtemp.shape))
    while True:
        try:
            #print("pyrUpimg: "+str(type(cv2.pyrUp(img))))
            img = cv2.pyrUp(img.astype('float32')) + (image_pyramid.pop())
        except:
            break
    #print("Collapse done!"+str(img.shape))
    return img.astype('float16')


def collapse_laplacian_video_pyramid(pyramid):
    i = 0
    pyd = []
    while True:
        try:
            img_pyramid = [vid[i] for vid in pyramid]
            pyd.append(collapse_laplacian_pyramid(img_pyramid))
            i += 1
        except IndexError:
            break
    print("pyd: "+str(pyd[0].shape))
    return pyd
