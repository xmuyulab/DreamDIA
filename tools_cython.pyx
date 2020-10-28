import numpy as np
cimport numpy as np

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function


cdef _smooth_array(np.ndarray[np.float_t, ndim=2] arr):
    if arr.shape[1] <= 1:
        return arr

    cdef int xmax = arr.shape[0]
    cdef int ymax = arr.shape[1]

    cdef np.ndarray[np.float_t,ndim=2] new_arr = np.zeros([xmax, ymax], dtype=np.float)
    
    cdef int x, y

    for x in range(xmax):
        new_arr[x,0] = (2*arr[x,0]+arr[x,1])/3
        new_arr[x,ymax-1] = (2*arr[x,ymax-1]+arr[x,ymax-2])/3
        for y in range(1, ymax-1):
            new_arr[x,y] = 0.5*arr[x,y]+0.25*(arr[x,y-1]+arr[x,y+1])

    return new_arr

def smooth_array(arr):
    return _smooth_array(arr)

cdef _calc_area(np.ndarray[np.float_t, ndim=1] frag_chrom, np.ndarray[np.float_t, ndim=1] rt_list_diff):
    cdef float trapezoids = 0
    cdef int i
    for i in range(len(rt_list_diff)):
        trapezoids += (frag_chrom[i] + frag_chrom[i + 1]) * rt_list_diff[i]
    return trapezoids / 2

def calc_area(frag_chrom, rt_list_diff):
    return _calc_area(frag_chrom, rt_list_diff)
