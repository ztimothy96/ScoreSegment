import matplotlib.pyplot as plt
import numpy as np
import cv2
import streamlit as st

def plot_histogram(x, thresh):
    if not st.session_state['debug']:
        return
    fig, _ = plt.subplots()
    plt.hist(x)
    plt.axvline(x=thresh, color="black", linestyle="--")
    st.pyplot(fig)

def plot_lines(im, lines, axis=0):
    if not st.session_state['debug']:
        return
    fig, _ = plt.subplots()
    if axis==0:
        W = im.shape[1]
        for i in range(len(lines)):
            x = [0, W-1]
            y = [lines[i], lines[i]]
            plt.plot(x, y, color='blue', linewidth=0.5)
    elif axis==1:
        H = im.shape[0]
        for i in range(len(lines)):
            x = [lines[i], lines[i]]
            y = [0, H-1]
            plt.plot(x, y, color='green', linewidth=0.5)
    else:
        st.error('Invalid axis')
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    st.pyplot(fig)

def find_staff_lines(im_bin):
    brightness = np.mean(im_bin, axis=1)
    sorted_brightness = np.sort(brightness)
    sorted_diff = np.diff(sorted_brightness)
    i = np.argmax(sorted_diff)
    thresh = sorted_brightness[i] + sorted_diff[i]/2
    staff_lines = (brightness < thresh).nonzero()[0]
    plot_histogram(sorted_brightness, thresh)
    plot_lines(im_bin, staff_lines, axis=0)
    return staff_lines

def find_staff_borders(im_bin, window = 50):
    H = im_bin.shape[0]
    staff_lines = find_staff_lines(im_bin)
    brightness = np.mean(im_bin, axis=0)
    # get the x coordinate at the start of the staves
    # this can actually be the end of the stave if the first stave is offset...
    start_x = np.argmin(brightness)
    plot_lines(im_bin, [start_x], axis=1)

    # get the y coordinates at the top and bottom of each system 
    dif = im_bin[1:H, start_x] - im_bin[:H-1, start_x]
    dif_lines = (dif != 0).nonzero()[0] 
    plot_lines(im_bin, dif_lines, axis=0)

    border_lines = []
    for dif_line in dif_lines:
        closest = np.argmin((staff_lines - dif_line)**2)
        if (staff_lines[closest] - dif_line)**2 < window**2:
            border_lines.append(staff_lines[closest])
    plot_lines(im_bin, border_lines, axis=0)
    return np.unique(border_lines)

def find_seps(im_bin, border_lines):
    seps = []
    i = 1
    while i < len(border_lines)-1:
        e, s = border_lines[i], border_lines[i+1]
        dark = 1 - np.mean(im_bin[e: s], axis=1)
        dist = (np.arange(e, s) - (e + s)//2)**2 / ((e+s)//2)**2
        seps.append(e+np.argmin(dark + dist))
        i += 2
    return seps

# input: PIL image
def get_staves(im):
    thresh = 0.5
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    im_bin = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im_bin = cv2.threshold(im_bin*1.0/255, thresh, 1.0, cv2.THRESH_BINARY)[1]

    border_lines = find_staff_borders(im_bin)
    plot_lines(im_bin, border_lines, axis=0)
    seps = find_seps(im_bin, border_lines)
    plot_lines(im_bin, seps, axis=0)

    staves = np.split(im, seps, axis=0)
    return staves
