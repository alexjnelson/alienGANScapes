import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.function_base import average


path = 'other datasets/_pixabay dataset colored'
rng = np.random.default_rng(seed=0)


for file in os.listdir(path)[:5]:
    im = Image.open(os.path.join(path, file))
    data = np.array(im)

    sample_rows = int(data.shape[1] * 0.3)

    mean = np.mean(data[:, :sample_rows, :])
    stdev = np.std(data[:, :sample_rows, :])
    tgt_color = rng.normal(mean, stdev, 3)

    # sample the top 30% rows of the image
    # get the average color of the sky
    # change the average color to the target color
    Rs, Gs, Bs = [], [], []
    for row in data[:, :sample_rows, :]:
        for p in row:
            Rs.append(p[0])
            Gs.append(p[1])
            Bs.append(p[2])

    avg_R, avg_G, avg_B = average(Rs), average(Gs), average(Bs)
    avg_color = (avg_R, avg_G, avg_B)

    for col in data:
        for p in col:
            dif_R, dif_G, dif_B = abs(p[0] - avg_R), abs(p[1] - avg_G), abs(p[2] - avg_B)
            adj_R, adj_G, adj_B = 255 - dif_R, 255 - dif_G, 255 - dif_B

            frac_R, frac_G, frac_B = adj_R / 255, adj_G / 255, adj_B / 255

            p[0] -= min(avg_R * frac_R, p[0])
            p[1] -= min(avg_G * frac_G, p[1])
            p[2] -= min(avg_B * frac_B, p[2])

            p[0] += min(tgt_color[0] * frac_R, 255 - p[0])
            p[1] += min(tgt_color[1] * frac_G, 255 - p[1])
            p[2] += min(tgt_color[2] * frac_B, 255 - p[2])

    plt.imshow(data)
    plt.axis('off')
    # plt.imsave(os.path.join(path, file))
    plt.show()
