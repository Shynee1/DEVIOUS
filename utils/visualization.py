import torch
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy
import os
import utils.helper_functions as helper
from matplotlib.lines import Line2D

def visualize_optical_flow(flow, savepath=None, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    if flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)
    flow[numpy.isinf(flow)]=0
    # Use Hue, Saturation, Value colour model
    hsv = numpy.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = numpy.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = numpy.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=numpy.pi*2
    hsv[..., 0] = ang/numpy.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    if scaling is None:
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    else:
        mag[mag>scaling]=scaling
        hsv[...,2] = mag/scaling
    rgb = colors.hsv_to_rgb(hsv)
    # This all seems like an overkill, but it's just to exactly match the cv2 implementation
    bgr = numpy.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)
    plot_with_pyplot = True
    if plot_with_pyplot:
        fig = plt.figure(frameon=False)
        plot = plt.imshow(bgr)
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
    if text is not None:
        plt.text(0, -5, text)

    if savepath is not None:
        if plot_with_pyplot:
            fig.savefig(savepath, bbox_inches='tight', dpi=200)
            plt.close()
    return bgr, (mag.min(), mag.max())


def visualize_optical_flow_arrows(flow, savepath=None, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    if flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)
    flow[numpy.isinf(flow)] = 0
    
    h, w = flow.shape[:2]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.axis('off')

    flow = flow * 1032
    
    # Calculate magnitude and angle
    mag = numpy.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    print("Magnitude min:", mag.min())
    print("Magnitude max:", mag.max())
    print("Magnitude mean:", mag.mean())

    # Apply scaling if provided
    if scaling is not None:
        mag_display = mag
        scale_factor = scaling
    else:
        mag_display = mag
        scale_factor = mag.max() if mag.max() > 0 else 1
    
    # Downsample for visualization (draw arrows every N pixels)
    step = max(1, min(h, w) // 40)  # Adaptive step size based on image size
    y_coords, x_coords = numpy.mgrid[step//2:h:step, step//2:w:step]
    
    # Get flow vectors at sampled points
    u = flow[y_coords, x_coords, 0]
    v = flow[y_coords, x_coords, 1]
    mag_sampled = mag_display[y_coords, x_coords]
    
    # Normalize arrow lengths
    arrow_scale = step * 0.8  # Scale arrows to fit grid
    if scale_factor > 0:
        u_norm = u / scale_factor * arrow_scale
        v_norm = v / scale_factor * arrow_scale
    else:
        u_norm = u * 0
        v_norm = v * 0
    
    # Create color map based on magnitude
    if scale_factor > 0:
        colors_norm = mag_sampled / scale_factor
    else:
        colors_norm = numpy.zeros_like(mag_sampled)
    
    # Draw arrows
    for i in range(y_coords.shape[0]):
        for j in range(y_coords.shape[1]):
            if mag_sampled[i, j] > 0.1:  # Only draw arrows for significant flow
                # Color based on magnitude (blue to red)
                color_intensity = colors_norm[i, j]
                color = plt.cm.jet(color_intensity)
                
                ax.arrow(x_coords[i, j], y_coords[i, j], 
                        u_norm[i, j], v_norm[i, j],
                        head_width=step*(1/9), head_length=step*(1/3),
                        fc=color, ec=color, alpha=0.8,
                        length_includes_head=True)
    
    # Add text if provided
    if text is not None:
        ax.text(5, 15, text, fontsize=12, color='white', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    # Set background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=200, facecolor='black')
        plt.close()
    
    if return_image:
        # Convert figure to numpy array
        fig.canvas.draw()
        buf = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return buf, (mag.min(), mag.max())
    else:
        plt.close()
        return None, (mag.min(), mag.max())
