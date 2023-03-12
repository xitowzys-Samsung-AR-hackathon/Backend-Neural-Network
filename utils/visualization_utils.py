from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def visualise_keypoints(im1, im2, pts, conf1=None, conf2=None):
    new_cmap = rand_cmap(100, type='bright', first_color_black=True, last_color_black=False, verbose=False)

    pts = [pts[0], pts[1]]
    def draw_keypoints(im, pts, imW=im1.size[0]):
        ptsA, ptsB = pts
        draw = ImageDraw.Draw(im)
        r = 4
        for j in range(0, ptsA.shape[0]):
            draw.ellipse([ptsA[j,0] - r, ptsA[j,1] - r, ptsA[j,0] + r, ptsA[j,1] + r], fill=(255,0,0,255))
            draw.ellipse([ptsB[j,0] - r + imW, ptsB[j,1] - r, ptsB[j,0] + r + imW, ptsB[j,1] + r], fill=(255,0,0,255))
            R, G, B, A = new_cmap(j)
            draw.line([ptsA[j,0], ptsA[j,1], ptsB[j,0] + imW, ptsB[j,1]], width=r, fill=(int(R*255), 
                                                                                         int(G*255), 
                                                                                         int(B*255), 
                                                                                         int(A*255)))
              
    fig = plt.figure(figsize=(30,15))
    axes = fig.subplots(nrows=1,ncols=1)    
    
    # Pad to make the heights the same
    if im1.size[1] < im2.size[1]:
        W_n, H_n = im1.size
        W_2, H_2 = im2.size
        A = H_2 / H_n
        
        H_n = H_2
        W_n = int(A * W_n)
        
        im1 = im1.resize((W_n, H_n))
        pts[0] = pts[0] * A
    elif im1.size[1] > im2.size[1]:
        W_1, H_1 = im1.size
        W_n, H_n = im2.size
        A = H_1 / H_n

        H_n = H_1
        W_n = int(A * W_n)
        im2 = im2.resize((W_n, H_n))
        pts[1] = pts[1] * A
    
        
    im1 = np.array(im1)
    im2 = np.array(im2)
    
    im = Image.fromarray(np.hstack((im1, im2)))
    draw_keypoints(im, pts, imW=im1.shape[1])
    axes.imshow(im)
    
    plt.axis('off')
    
    return im