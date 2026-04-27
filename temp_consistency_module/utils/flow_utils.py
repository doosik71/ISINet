import numpy as np

TAG_CHAR = np.array([202021.25], np.float32)


UNKNOWN_FLOW_THRESH = 1e7


def make_color_wheel():
    """Build a Middlebury-style color wheel for optical flow visualization."""
    ry = 15
    yg = 6
    gc = 4
    cb = 11
    bm = 13
    mr = 6
    ncols = ry + yg + gc + cb + bm + mr

    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0

    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    col += ry

    colorwheel[col:col + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[col:col + yg, 1] = 255
    col += yg

    colorwheel[col:col + gc, 1] = 255
    colorwheel[col:col + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    col += gc

    colorwheel[col:col + cb, 1] = 255 - np.floor(255 * np.arange(0, cb) / cb)
    colorwheel[col:col + cb, 2] = 255
    col += cb

    colorwheel[col:col + bm, 2] = 255
    colorwheel[col:col + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    col += bm

    colorwheel[col:col + mr, 2] = 255 - np.floor(255 * np.arange(0, mr) / mr)
    colorwheel[col:col + mr, 0] = 255

    return colorwheel


def compute_color(u, v):
    """Map normalized horizontal/vertical flow to an RGB image."""
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(-v, -u) / np.pi

    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    img = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)

    for channel in range(3):
        col0 = colorwheel[k0, channel] / 255.0
        col1 = colorwheel[k1, channel] / 255.0
        col = (1 - f) * col0 + f * col1

        inside = rad <= 1
        col[inside] = 1 - rad[inside] * (1 - col[inside])
        col[~inside] *= 0.75

        img[:, :, channel] = np.floor(255 * col).astype(np.uint8)

    return img


def flow_to_image(flow):
    """Convert a HxWx2 flow map to a color visualization image."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Expected flow with shape (H, W, 2)")

    u = flow[:, :, 0].copy()
    v = flow[:, :, 1].copy()

    invalid = (
        np.isnan(u) | np.isnan(v) |
        (np.abs(u) > UNKNOWN_FLOW_THRESH) |
        (np.abs(v) > UNKNOWN_FLOW_THRESH)
    )
    u[invalid] = 0
    v[invalid] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    max_rad = np.max(rad)
    if max_rad > 0:
        u /= max_rad
        v /= max_rad

    img = compute_color(u, v)
    img[invalid] = 0
    return img

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
