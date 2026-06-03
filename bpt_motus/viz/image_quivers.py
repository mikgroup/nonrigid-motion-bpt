# -*- coding: utf-8 -*-
"""This module contains plotting functions based on matplotlib
for image, line, and scatter plots.

A feature of these plotting functions is that
they can be controlled using only hotkeys
so the user does not need to move away from the keyboard.

Given an array ``x``, an example usage is:

    >>> ImagePlot(x)
    >>> LinePlot(x)
    >>> ScatterPlot(x)

"""
import datetime
import os
import subprocess
import uuid

import numpy as np

import sigpy as sp
 
__all__ = ["ImagePlot", "LinePlot", "ScatterPlot"]


image_plot_help_str = r"""
$\bf{Hotkeys:}$
    $\bf{h:}$ show/hide hotkey menu.
    $\bf{x/y/z:}$ set current axis as x/y/z.
    $\bf{t:}$ swap between x and y.
    $\bf{c:}$ select current axis as color.
    $\bf{left/right:}$ change current axis.
    $\bf{up/down:}$ change slice along current axis.
    $\bf{a:}$ toggle hide all labels, titles and axes.
    $\bf{m/p/r/i/l:}$  magnitude/phase/real/imaginary/log mode.
    $\bf{[/]:}$ change brightness.
    $\bf{\{/\}:}$ change contrast.
    $\bf{s:}$ save as png.
    $\bf{g/v:}$ save as gif/video by along current axis.
    $\bf{q:}$ refresh.
    $\bf{0-9:}$ enter slice number.
    $\bf{enter:}$ set current axis as slice number.
"""


class ImagePlot(object):
    """Plot array as image.

    Press 'h' for a menu for hotkeys.

    Args:
        im (array): image numpy/cupy array.
        x (int): x axis.
        y (int): y axis.
        z (None or int): z axis.
        c (None or int): color axis.
        hide_axes (bool): toggle hiding axes, labels and title.
        mode (str): specify magnitude, phase, real, imaginary,
            and log mode. {'m', 'p', 'r', 'i', 'l'}.
        title (str): title.
        interpolation (str): plot interpolation.
        save_basename (str): saved png, gif, and video base name.
        fps (int): frame per seconds for gif and video.

    """

    def __init__(
        self,
        im,
        x=-1,
        y=-2,
        z=None,
        c=None,
        hide_axes=False,
        mode=None,
        colormap=None,
        vmin=None,
        vmax=None,
        title="",
        interpolation="nearest",
        save_basename="Figure",
        fps=10,
    ):
        if im.ndim < 2:
            raise TypeError(
                "Image dimension must at least be two, got {im_ndim}".format(
                    im_ndim=im.ndim
                )
            )
        import matplotlib.pyplot as plt

        self.axim = None
        self.im = im
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.shape = self.im.shape
        self.ndim = self.im.ndim
        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.x = x % self.ndim
        self.y = y % self.ndim
        self.z = z % self.ndim if z is not None else None
        self.c = c % self.ndim if c is not None else None
        self.d = max(self.ndim - 3, 0)
        self.hide_axes = hide_axes
        self.show_help = False
        self.title = title
        self.interpolation = interpolation
        self.mode = mode
        self.colormap = colormap
        self.entering_slice = False
        self.vmin = vmin
        self.vmax = vmax
        self.save_basename = save_basename
        self.fps = fps
        self.help_text = None

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id
        )
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == "up":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "down":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] - 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "left":
            self.d = (self.d - 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "right":
            self.d = (self.d + 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "x" and self.d not in [self.x, self.z, self.c]:
            if self.d == self.y:
                self.x, self.y = self.y, self.x
            else:
                self.x = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "y" and self.d not in [self.y, self.z, self.c]:
            if self.d == self.x:
                self.x, self.y = self.y, self.x
            else:
                self.y = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "z" and self.d not in [self.x, self.y, self.c]:
            if self.d == self.z:
                self.z = None
            else:
                self.z = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif (
            event.key == "c"
            and self.d not in [self.x, self.y, self.z]
            and self.shape[self.d] == 3
        ):
            if self.d == self.c:
                self.c = None
            else:
                self.c = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "t":
            self.x, self.y = self.y, self.x

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "a":
            self.hide_axes = not self.hide_axes

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "f":
            self.fig.canvas.manager.full_screen_toggle()

        elif event.key == "q":
            self.vmin = None
            self.vmax = None
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "ö":
            width = self.vmax - self.vmin
            self.vmin -= width * 0.1
            self.vmax -= width * 0.1

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "ä":
            width = self.vmax - self.vmin
            self.vmin += width * 0.1
            self.vmax += width * 0.1

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "b":
            width = self.vmax - self.vmin
            center = (self.vmax + self.vmin) / 2
            self.vmin = center - width * 1.1 / 2
            self.vmax = center + width * 1.1 / 2

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "n":
            width = self.vmax - self.vmin
            center = (self.vmax + self.vmin) / 2
            self.vmin = center - width * 0.9 / 2
            self.vmax = center + width * 0.9 / 2

            self.update_image()
            self.fig.canvas.draw()

        elif event.key in ["m", "p", "r", "i", "l"]:
            self.vmin = None
            self.vmax = None
            self.mode = event.key

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "s":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.png"
            )
            self.fig.savefig(
                filename,
                transparent=True,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
            )

        elif event.key == "g":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.gif"
            )
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    bbox_inches=bbox,
                    pad_inches=0,
                    dpi=200,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "palettegen",
                    "{} palette.png".format(temp_basename),
                ]
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-i",
                    "{} palette.png".format(temp_basename),
                    "-lavfi",
                    "paletteuse",
                    filename,
                ]
            )

            os.remove("{} palette.png".format(temp_basename))
            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key == "v":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.mp4"
            )
            temp_basename = uuid.uuid4()

            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0.5,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-r",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "crop=floor(iw/2)*2-10:floor(ih/2)*2-10",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "1",
                    "-vcodec",
                    "libx264",
                    "-preset",
                    "veryslow",
                    filename,
                ]
            )

            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key in [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "backspace",
        ] and self.d not in [self.x, self.y, self.z, self.c]:
            if self.entering_slice:
                if event.key == "backspace":
                    if self.entered_slice < 10:
                        self.entering_slice = False
                    else:
                        self.entered_slice //= 10
                else:
                    self.entered_slice = self.entered_slice * 10 + int(
                        event.key
                    )
            elif event.key != "backspace":
                self.entering_slice = True
                self.entered_slice = int(event.key)

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "enter" and self.entering_slice:
            self.entering_slice = False
            if self.entered_slice < self.shape[self.d]:
                self.slices[self.d] = self.entered_slice

                self.update_image()

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "h":
            self.show_help = not self.show_help

            self.update_image()
            self.fig.canvas.draw()
        else:
            return

    def update_image(self):
        # Extract slice.
        idx = []
        for i in range(self.ndim):
            if i in [self.x, self.y, self.z, self.c]:
                idx.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])

        idx = tuple(idx)
        imv = sp.to_device(self.im[idx])

        # Transpose to have [z, y, x, c].
        imv_dims = [self.y, self.x]
        if self.z is not None:
            imv_dims = [self.z] + imv_dims

        if self.c is not None:
            imv_dims = imv_dims + [self.c]

        imv = np.transpose(imv, np.argsort(np.argsort(imv_dims)))
        imv = array_to_image(imv, color=self.c is not None)

        if self.mode is None:
            if np.isrealobj(imv):
                self.mode = "r"
            else:
                self.mode = "m"

        if self.mode == "m":
            imv = np.abs(imv)
        elif self.mode == "p":
            imv = np.angle(imv)
        elif self.mode == "r":
            imv = np.real(imv)
        elif self.mode == "i":
            imv = np.imag(imv)
        elif self.mode == "l":
            imv = np.abs(imv)
            imv = np.log(imv, out=np.ones_like(imv) * -31, where=imv != 0)

        if self.vmin is None:
            self.vmin = imv.min()

        if self.vmax is None:
            self.vmax = imv.max()

        if self.axim is None:
            if self.colormap is None:
                colormap = "gray"
            else:
                colormap = self.colormap
            self.axim = self.ax.imshow(
                imv,
                vmin=self.vmin,
                vmax=self.vmax,
                cmap=colormap,
                origin="lower",
                interpolation=self.interpolation,
                aspect=1.0,
                extent=[0, imv.shape[1], 0, imv.shape[0]],
            )

            if self.colormap is not None:
                self.fig.colorbar(self.axim)

        else:
            self.axim.set_data(imv)
            self.axim.set_extent([0, imv.shape[1], 0, imv.shape[0]])
            self.axim.set_clim(self.vmin, self.vmax)

        if self.help_text is None:
            bbox_props = dict(
                boxstyle="round", pad=1, fc="white", alpha=0.95, lw=0
            )
            self.help_text = self.ax.text(
                imv.shape[0] / 2,
                imv.shape[1] / 2,
                image_plot_help_str,
                ha="center",
                va="center",
                linespacing=1.5,
                ma="left",
                size=8,
                bbox=bbox_props,
            )

        self.help_text.set_visible(self.show_help)

    def update_axes(self):
        if not self.hide_axes:
            caption = "["
            for i in range(self.ndim):
                if i == self.d:
                    caption += "["
                else:
                    caption += " "

                if self.flips[i] == -1 and (
                    i == self.x or i == self.y or i == self.z or i == self.c
                ):
                    caption += "-"

                if i == self.x:
                    caption += "x"
                elif i == self.y:
                    caption += "y"
                elif i == self.z:
                    caption += "z"
                elif i == self.c:
                    caption += "c"
                elif i == self.d and self.entering_slice:
                    caption += str(self.entered_slice) + "_"
                else:
                    caption += str(self.slices[i])

                if i == self.d:
                    caption += "]"
                else:
                    caption += " "
            caption += "]"

            self.ax.set_title(caption)
            self.fig.suptitle(self.title)
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.title.set_visible(True)
        else:
            self.ax.set_title("")
            self.fig.suptitle("")
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            self.ax.title.set_visible(False)


def mosaic_shape(batch):
    mshape = [int(batch**0.5), batch // int(batch**0.5)]

    while sp.prod(mshape) < batch:
        mshape[1] += 1

    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1

    return tuple(mshape)


def array_to_image(arr, color=False):
    """
    Flattens all dimensions except the last two

    Args:
        arr (array): shape [z, x, y, c] if color, else [z, x, y]

    """
    if color and not (arr.max() == 0 and arr.min() == 0):
        arr = arr / np.abs(arr).max()

    if arr.ndim == 2:
        return arr
    elif color and arr.ndim == 3:
        return arr

    if color:
        img_shape = arr.shape[-3:]
        batch = sp.prod(arr.shape[:-3])
        mshape = mosaic_shape(batch)
    else:
        img_shape = arr.shape[-2:]
        batch = sp.prod(arr.shape[:-2])
        mshape = mosaic_shape(batch)

    if sp.prod(mshape) == batch:
        img = arr.reshape((batch,) + img_shape)
    else:
        img = np.zeros((sp.prod(mshape),) + img_shape, dtype=arr.dtype)
        img[:batch, ...] = arr.reshape((batch,) + img_shape)

    img = img.reshape(mshape + img_shape)
    if color:
        img = np.transpose(img, (0, 2, 1, 3, 4))
        img = img.reshape(
            (img_shape[0] * mshape[0], img_shape[1] * mshape[1], 3)
        )
    else:
        img = np.transpose(img, (0, 2, 1, 3))
        img = img.reshape((img_shape[0] * mshape[0], img_shape[1] * mshape[1]))

    return img


class LinePlot(object):
    """Plot array as lines.

    Keyword Args:
        x: select current dimension as x
        left/right: increment/decrement current dimension
        up/down: flip axis when current dimension is x or y
            otherwise increment/decrement slice at current dimension
        h: toggle hide all labels, titles and axes
        m: magnitude mode
        p: phase mode
        r: real mode
        i: imaginary mode
        l: log mode
        s: save as png.
        g: save as gif by traversing current dimension.
        v: save as video by traversing current dimension.
    """

    def __init__(
        self,
        arr,
        x=-1,
        hide_axes=False,
        mode="m",
        title="",
        save_basename="Figure",
        fps=10,
    ):
        import matplotlib.pyplot as plt

        self.arr = arr
        self.axarr = None

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.shape = self.arr.shape
        self.ndim = self.arr.ndim
        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.x = x % self.ndim
        self.d = max(self.ndim - 3, 0)
        self.hide_axes = hide_axes
        self.title = title
        self.mode = mode
        self.save_basename = save_basename
        self.fps = fps
        self.bottom = None
        self.top = None

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id
        )
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.update_axes()
        self.update_line()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == "up":
            if self.d != self.x:
                self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_line()
            self.fig.canvas.draw()

        elif event.key == "down":
            if self.d != self.x:
                self.slices[self.d] = (self.slices[self.d] - 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_line()
            self.fig.canvas.draw()

        elif event.key == "left":
            self.d = (self.d - 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "right":
            self.d = (self.d + 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "x" and self.d != self.x:
            self.x = self.d

            self.update_axes()
            self.update_line()
            self.fig.canvas.draw()

        elif event.key == "a":
            self.hide_axes = not self.hide_axes

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "f":
            self.fig.canvas.manager.full_screen_toggle()

        elif (
            event.key == "m"
            or event.key == "p"
            or event.key == "r"
            or event.key == "i"
            or event.key == "l"
        ):
            self.mode = event.key
            self.bottom = None
            self.top = None

            self.update_axes()
            self.update_line()
            self.fig.canvas.draw()

        elif event.key == "s":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.png"
            )
            self.fig.savefig(
                filename,
                transparent=True,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
            )

        elif event.key == "g":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.gif"
            )
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_line()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    bbox_inches=bbox,
                    pad_inches=0,
                    dpi=200,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "palettegen",
                    "{} palette.png".format(temp_basename),
                ]
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-i",
                    "{} palette.png".format(temp_basename),
                    "-lavfi",
                    "paletteuse",
                    filename,
                ]
            )

            print("Expected GIF output:", filename)
            print("Absolute path:", os.path.abspath(filename))
            print("Current working directory:", os.getcwd())

            os.remove("{} palette.png".format(temp_basename))
            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key == "v":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %h.%M.%S %p.mov"
            )
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_line()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    bbox_inches=bbox,
                    pad_inches=0,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vcodec",
                    "png",
                    filename,
                ]
            )

            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))
        else:
            return

        return

    def update_line(self):
        order = [i for i in range(self.ndim) if i != self.x] + [self.x]
        idx = tuple(
            [self.slices[i] for i in order[:-1]]
            + [slice(None, None, self.flips[self.x])]
        )

        arrv = self.arr.transpose(order)[idx]

        if self.mode == "m":
            arrv = np.abs(arrv)
        elif self.mode == "p":
            arrv = np.angle(arrv)
        elif self.mode == "r":
            arrv = np.real(arrv)
        elif self.mode == "i":
            arrv = np.imag(arrv)
        elif self.mode == "l":
            eps = 1e-31
            arrv = np.log(np.abs(arrv) + eps)

        if self.bottom is None:
            self.bottom = arrv.min()

        if self.top is None:
            self.top = arrv.max()

        if self.axarr is None:
            self.axarr = self.ax.plot(arrv)[0]

        else:
            self.axarr.set_xdata(np.arange(len(arrv)))
            self.axarr.set_ydata(arrv)
            self.ax.set_ylim(self.bottom, self.top)

    def update_axes(self):
        if not self.hide_axes:
            caption = "Slice: ["
            for i in range(self.ndim):
                if i == self.d:
                    caption += "["
                else:
                    caption += " "

                if self.flips[i] == -1 and i == self.x:
                    caption += "-"

                if i == self.x:
                    caption += "x"
                else:
                    caption += str(self.slices[i])

                if i == self.d:
                    caption += "]"
                else:
                    caption += " "
            caption += "]"

            self.ax.set_title(caption)
            self.ax.axis("on")
            self.fig.suptitle(self.title)
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.title.set_visible(True)
        else:
            self.ax.set_title("")
            self.fig.suptitle("")
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            self.ax.title.set_visible(False)


class ScatterPlot(object):
    """Plot array as scatter.

    Keyword Args:
        z: toggle current dimension as z dimension
        left/right: increment/decrement current dimension
        up/down: flip axis when current dimension is x or y
            otherwise increment/decrement slice at current dimension
        h: toggle hide all labels, titles and axes
        m: magnitude mode
        p: phase mode
        r: real mode
        i: imaginary mode
        l: log mode
    """

    def __init__(
        self,
        coord,
        data=None,
        z=None,
        hide_axes=False,
        mode="m",
        title="",
        save_basename="Figure",
        fps=10,
    ):
        import matplotlib.pyplot as plt

        self.coord = coord
        assert coord.shape[-1] == 2
        if data is None:
            self.data = np.ones(coord.shape[:-1])
        else:
            self.data = data

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("k")
        self.ax.axis("equal")

        for c, d in zip(coord.shape[:-1], self.data.shape[-coord.ndim + 1 :]):
            assert c == d

        self.ndim = self.data.ndim - self.coord.ndim + 1
        self.shape = self.data.shape[: self.ndim]

        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.z = z % self.ndim if z is not None else None
        self.d = 0
        self.hide_axes = hide_axes
        self.title = title
        self.mode = mode
        self.axsc = None
        self.entering_slice = False
        self.save_basename = save_basename
        self.fps = fps
        self.vmin = None
        self.vmax = None

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id
        )
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.update_axes()
        self.update_data()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == "up":
            if self.d != self.z:
                self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_data()
            self.fig.canvas.draw()

        elif event.key == "down":
            if self.d != self.z:
                self.slices[self.d] = (self.slices[self.d] - 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_data()
            self.fig.canvas.draw()

        elif event.key == "left":
            self.d = (self.d - 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "right":
            self.d = (self.d + 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        # elif event.key == 'z':
        #     if self.d == self.z:
        #         self.z = None
        #     else:
        #         self.z = self.d

        #     self.update_axes()
        #     self.update_data()
        #     self.fig.canvas.draw()

        elif event.key == "a":
            self.hide_axes = not self.hide_axes

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "f":
            self.fig.canvas.manager.full_screen_toggle()

        elif (
            event.key == "m"
            or event.key == "p"
            or event.key == "r"
            or event.key == "i"
            or event.key == "l"
        ):
            self.mode = event.key
            self.vmin = None
            self.vmax = None

            self.update_axes()
            self.update_data()
            self.fig.canvas.draw()

        elif event.key == "s":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.png"
            )
            self.fig.savefig(
                filename,
                transparent=True,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
            )

        elif event.key == "g":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.gif"
            )
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_data()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    bbox_inches=bbox,
                    pad_inches=0,
                    dpi=200,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "palettegen",
                    "{} palette.png".format(temp_basename),
                ]
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-i",
                    "{} palette.png".format(temp_basename),
                    "-lavfi",
                    "paletteuse",
                    filename,
                ]
            )

            os.remove("{} palette.png".format(temp_basename))
            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key == "v":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %h.%M.%S %p.mov"
            )
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_data()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    bbox_inches=bbox,
                    pad_inches=0,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vcodec",
                    "png",
                    filename,
                ]
            )

            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif (
            event.key
            in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "backspace"]
            and self.d != self.z
        ):
            if self.entering_slice:
                if event.key == "backspace":
                    if self.entered_slice < 10:
                        self.entering_slice = False
                    else:
                        self.entered_slice //= 10
                else:
                    self.entered_slice = self.entered_slice * 10 + int(
                        event.key
                    )
            else:
                self.entering_slice = True
                self.entered_slice = int(event.key)

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "enter" and self.entering_slice:
            self.entering_slice = False
            if self.entered_slice < self.shape[self.d]:
                self.slices[self.d] = self.entered_slice

                self.update_data()

            self.update_axes()
            self.fig.canvas.draw()

        else:
            return

    def update_data(self):
        idx = []
        for i in range(self.ndim):
            if i == self.z:
                idx.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])

        idx = tuple(idx)
        if idx:
            datav = sp.to_device(self.data[idx])
        else:
            datav = sp.to_device(self.data)

        # if self.z is not None:
        #     datav_dims = [self.z] + datav_dims
        coordv = sp.to_device(self.coord)

        if self.mode == "m":
            datav = np.abs(datav)
        elif self.mode == "p":
            datav = np.angle(datav)
        elif self.mode == "r":
            datav = np.real(datav)
        elif self.mode == "i":
            datav = np.imag(datav)
        elif self.mode == "l":
            eps = 1e-31
            datav = np.log(np.abs(datav) + eps)

        datav = datav.ravel()
        if self.vmin is None:
            if datav.min() == datav.max():
                self.vmin = 0
            else:
                self.vmin = datav.min()

        if self.vmax is None:
            self.vmax = datav.max()

        if self.axsc is None:
            self.axsc = self.ax.scatter(
                coordv[..., 0].ravel(),
                coordv[..., 1].ravel(),
                c=datav,
                s=1,
                linewidths=0,
                cmap="gray",
                vmin=self.vmin,
                vmax=self.vmax,
            )

        else:
            self.axsc.set_offsets(coordv.T.reshape([-1, 2]))
            self.axsc.set_color(datav)

    def update_axes(self):
        if not self.hide_axes:
            caption = "["
            for i in range(self.ndim):
                if i == self.d:
                    caption += "["
                else:
                    caption += " "

                if self.flips[i] == -1 and i == self.z:
                    caption += "-"

                if i == self.z:
                    caption += "z"
                elif i == self.d and self.entering_slice:
                    caption += str(self.entered_slice) + "_"
                else:
                    caption += str(self.slices[i])

                if i == self.d:
                    caption += "]"
                else:
                    caption += " "
            caption += "]"

            self.ax.set_title(caption)
            self.fig.suptitle(self.title)
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.title.set_visible(True)
        else:
            self.ax.set_title("")
            self.fig.suptitle("")
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            self.ax.title.set_visible(False)

class QuiverPlot(object):
    """Plot first array as image, and second array as quiver plot on top of it.

    Press 'h' for a menu for hotkeys.

    Args:
        im (array): image numpy/cupy array.
        dvf (array): vector field numpy/cupy array.
        x (int): x axis.
        y (int): y axis.
        z (None or int): z axis.
        c (None or int): color axis.
        hide_axes (bool): toggle hiding axes, labels and title.
        mode (str): specify magnitude, phase, real, imaginary,
            and log mode. {'m', 'p', 'r', 'i', 'l'}.
        title (str): title.
        interpolation (str): plot interpolation.
        save_basename (str): saved png, gif, and video base name.
        fps (int): frame per seconds for gif and video.

    """

    def __init__(
        self,
        im,
        dvf,
        x=-1,
        y=-2,
        z=None,
        c=None,
        hide_axes=False,
        mode=None,
        colormap=None,
        vmin=None,
        vmax=None,
        title="",
        interpolation="nearest",
        save_basename="Figure",
        fps=10,
        quiver_scale=5e-2,
        quiver_step=2,
        quiver_color='g',
    ):
        if im.ndim < 2:
            raise TypeError(
                "Image dimension must at least be two, got {im_ndim}".format(
                    im_ndim=im.ndim
                )
            )


        import matplotlib.pyplot as plt

        self.axim = None
        self.axquiver = None
        self.im = im
        self.dvf = dvf
        self.quiver_scale = quiver_scale 
        self.quiver_step = quiver_step
        self.quiver_color = quiver_color
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.shape = self.im.shape
        self.ndim = self.im.ndim
        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.x = x % self.ndim
        self.y = y % self.ndim
        self.z = z % self.ndim if z is not None else None
        self.c = c % self.ndim if c is not None else None
        self.d = max(self.ndim - 3, 0)
        self.hide_axes = hide_axes
        self.show_help = False
        self.title = title
        self.interpolation = interpolation
        self.mode = mode
        self.colormap = colormap
        self.entering_slice = False
        self.vmin = vmin
        self.vmax = vmax
        self.save_basename = save_basename
        self.fps = fps
        self.help_text = None

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id
        )
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == "up":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "down":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] - 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "left":
            self.d = (self.d - 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "right":
            self.d = (self.d + 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "x" and self.d not in [self.x, self.z, self.c]:
            if self.d == self.y:
                self.x, self.y = self.y, self.x
            else:
                self.x = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "y" and self.d not in [self.y, self.z, self.c]:
            if self.d == self.x:
                self.x, self.y = self.y, self.x
            else:
                self.y = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "z" and self.d not in [self.x, self.y, self.c]:
            if self.d == self.z:
                self.z = None
            else:
                self.z = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif (
            event.key == "c"
            and self.d not in [self.x, self.y, self.z]
            and self.shape[self.d] == 3
        ):
            if self.d == self.c:
                self.c = None
            else:
                self.c = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "t":
            self.x, self.y = self.y, self.x

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "a":
            self.hide_axes = not self.hide_axes

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "f":
            self.fig.canvas.manager.full_screen_toggle()

        elif event.key == "q":
            self.vmin = None
            self.vmax = None
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "]":
            width = self.vmax - self.vmin
            self.vmin -= width * 0.1
            self.vmax -= width * 0.1

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "[":
            width = self.vmax - self.vmin
            self.vmin += width * 0.1
            self.vmax += width * 0.1

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "}":
            width = self.vmax - self.vmin
            center = (self.vmax + self.vmin) / 2
            self.vmin = center - width * 1.1 / 2
            self.vmax = center + width * 1.1 / 2

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "{":
            width = self.vmax - self.vmin
            center = (self.vmax + self.vmin) / 2
            self.vmin = center - width * 0.9 / 2
            self.vmax = center + width * 0.9 / 2

            self.update_image()
            self.fig.canvas.draw()

        elif event.key in ["m", "p", "r", "i", "l"]:
            self.vmin = None
            self.vmax = None
            self.mode = event.key

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "s":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.png"
            )
            self.fig.savefig(
                filename,
                transparent=True,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
            )

        elif event.key == "k":
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i
                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()

        elif event.key == "g":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.gif"
            )
            print(filename)
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    bbox_inches=bbox,
                    pad_inches=0,
                    dpi=200,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "palettegen",
                    "{} palette.png".format(temp_basename),
                ]
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-i",
                    "{} palette.png".format(temp_basename),
                    "-lavfi",
                    "paletteuse",
                    filename,
                ]
            )

            os.remove("{} palette.png".format(temp_basename))
            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key == "v":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.mp4"
            )
            temp_basename = uuid.uuid4()

            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-r",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "crop=floor(iw/2)*2-10:floor(ih/2)*2-10",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "1",
                    "-vcodec",
                    "libx264",
                    "-preset",
                    "veryslow",
                    filename,
                ]
            )

            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key in [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "backspace",
        ] and self.d not in [self.x, self.y, self.z, self.c]:
            if self.entering_slice:
                if event.key == "backspace":
                    if self.entered_slice < 10:
                        self.entering_slice = False
                    else:
                        self.entered_slice //= 10
                else:
                    self.entered_slice = self.entered_slice * 10 + int(
                        event.key
                    )
            elif event.key != "backspace":
                self.entering_slice = True
                self.entered_slice = int(event.key)

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "enter" and self.entering_slice:
            self.entering_slice = False
            if self.entered_slice < self.shape[self.d]:
                self.slices[self.d] = self.entered_slice

                self.update_image()

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "h":
            self.show_help = not self.show_help

            self.update_image()
            self.fig.canvas.draw()
        else:
            return

    def update_image(self):
        # Extract slice.
        idx = []
        dvf_idx_horizontal = []
        dvf_idx_vertical = []
        for i in range(self.ndim):
            if i in [self.x, self.y, self.z, self.c]:
                idx.append(slice(None, None, self.flips[i]))
                dvf_idx_horizontal.append(slice(None, None, self.flips[i]))
                dvf_idx_vertical.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])
                dvf_idx_horizontal.append(self.slices[i])
                dvf_idx_vertical.append(self.slices[i])

        if len(self.dvf.shape) <= 4:
            dvf_idx_horizontal.append(self.x)
            dvf_idx_vertical.append(self.y)
        elif len(self.dvf.shape) == 5:
            dvf_idx_horizontal.append(self.x-int(1))
            dvf_idx_vertical.append(self.y-int(1))
        idx = tuple(idx)
        dvf_idx_horizontal = tuple(dvf_idx_horizontal)
        dvf_idx_vertical = tuple(dvf_idx_vertical)
        imv = sp.to_device(self.im[idx])
        dvf_display_x = sp.to_device(self.dvf[dvf_idx_horizontal]*self.flips[self.x])
        dvf_display_y = sp.to_device(self.dvf[dvf_idx_vertical]*self.flips[self.y])

        
        # Transpose to have [z, y, x, c].
        imv_dims = [self.y, self.x]
        if self.z is not None:
            imv_dims = [self.z] + imv_dims

        if self.c is not None:
            imv_dims = imv_dims + [self.c]
        
        imv = np.transpose(imv, np.argsort(np.argsort(imv_dims)))
        dvf_display_x = np.transpose(dvf_display_x, np.argsort(np.argsort(imv_dims)))
        dvf_display_y = np.transpose(dvf_display_y, np.argsort(np.argsort(imv_dims)))
        imv = array_to_image(imv, color=self.c is not None)
        dvf_grid = np.meshgrid(np.arange(imv.shape[1]),np.arange(imv.shape[0]))

        if self.mode is None:
            if np.isrealobj(imv):
                self.mode = "r"
            else:
                self.mode = "m"

        if self.mode == "m":
            imv = np.abs(imv)
        elif self.mode == "p":
            imv = np.angle(imv)
        elif self.mode == "r":
            imv = np.real(imv)
        elif self.mode == "i":
            imv = np.imag(imv)
        elif self.mode == "l":
            imv = np.abs(imv)
            imv = np.log(imv, out=np.ones_like(imv) * -31, where=imv != 0)

        if self.vmin is None:
            self.vmin = imv.min()

        if self.vmax is None:
            self.vmax = imv.max()

        if self.axquiver is None:
            self.axquiver = self.ax.quiver(
                dvf_grid[0][::self.quiver_step,::self.quiver_step],
                dvf_grid[1][::self.quiver_step,::self.quiver_step], 
                dvf_display_x[::self.quiver_step,::self.quiver_step],
                dvf_display_y[::self.quiver_step,::self.quiver_step],
                scale=self.quiver_scale, # higher is smaller arrows
                color='tab:orange', # TODO: make it 'g',
                width=0.01,      # ADD/CHANGE: Set the arrow shaft width (a good starting value)
                headlength=5,     # CHANGE: Restore or increase head length (default is 5)
                headwidth=3,      # CHANGE: Restore or increase head width (default is 3)
                minlength=1,
                # headlength=0, # remove green dots for 0 MFs
                # headwidth=0, # remove green dots for 0 MFs
                # minlength=0, # remove green dots for 0 MFs
            )
        else:
            self.axquiver.remove() 
            self.axquiver = self.ax.quiver(
                dvf_grid[0][::self.quiver_step,::self.quiver_step],
                dvf_grid[1][::self.quiver_step,::self.quiver_step], 
                dvf_display_x[::self.quiver_step,::self.quiver_step],
                dvf_display_y[::self.quiver_step,::self.quiver_step],
                scale=self.quiver_scale, # higher is smaller arrows
                color='tab:orange', # TODO: make it 'g'
                width=0.01,      # ADD/CHANGE: Set the arrow shaft width (a good starting value)
                headlength=5,     # CHANGE: Restore or increase head length (default is 5)
                headwidth=3,      # CHANGE: Restore or increase head width (default is 3)
                minlength=1,
                # headlength=0, # remove green dots for 0 MFs
                # headwidth=0, # remove green dots for 0 MFs
                # minlength=0, # remove green dots for 0 MFs
            )
        
        if self.axim is None:
            if self.colormap is None:
                colormap = "gray"
            else:
                colormap = self.colormap
            self.axim = self.ax.imshow(
                imv,
                vmin=self.vmin,
                vmax=self.vmax,
                cmap=colormap,
                origin="lower",
                interpolation=self.interpolation,
                aspect=1.0,
                extent=[0, imv.shape[1], 0, imv.shape[0]],
            )
            

            if self.colormap is not None:
                self.fig.colorbar(self.axim)

        else:
            self.axim.set_data(imv)
            self.axim.set_extent([0, imv.shape[1], 0, imv.shape[0]])
            self.axim.set_clim(self.vmin, self.vmax)

        if self.help_text is None:
            bbox_props = dict(
                boxstyle="round", pad=1, fc="white", alpha=0.95, lw=0
            )
            self.help_text = self.ax.text(
                imv.shape[0] / 2,
                imv.shape[1] / 2,
                image_plot_help_str,
                ha="center",
                va="center",
                linespacing=1.5,
                ma="left",
                size=8,
                bbox=bbox_props,
            )

        self.help_text.set_visible(self.show_help)

    def update_axes(self):
        if not self.hide_axes:
            caption = "["
            for i in range(self.ndim):
                if i == self.d:
                    caption += "["
                else:
                    caption += " "

                if self.flips[i] == -1 and (
                    i == self.x or i == self.y or i == self.z or i == self.c
                ):
                    caption += "-"

                if i == self.x:
                    caption += "x"
                elif i == self.y:
                    caption += "y"
                elif i == self.z:
                    caption += "z"
                elif i == self.c:
                    caption += "c"
                elif i == self.d and self.entering_slice:
                    caption += str(self.entered_slice) + "_"
                else:
                    caption += str(self.slices[i])

                if i == self.d:
                    caption += "]"
                else:
                    caption += " "
            caption += "]"

            self.ax.set_title(caption)
            self.fig.suptitle(self.title)
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.title.set_visible(True)
        else:
            self.ax.set_title("")
            self.fig.suptitle("")
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            self.ax.title.set_visible(False)

class QuiverAndImagePlot(object):
    """Plot first array as image, and second array as quiver plot next to it.

    Press 'h' for a menu for hotkeys.

    Args:
        im (array): image numpy/cupy array.
        dvf (array): vector field numpy/cupy array.
        x (int): x axis.
        y (int): y axis.
        z (None or int): z axis.
        c (None or int): color axis.
        hide_axes (bool): toggle hiding axes, labels and title.
        mode (str): specify magnitude, phase, real, imaginary,
            and log mode. {'m', 'p', 'r', 'i', 'l'}.
        title (str): title.
        interpolation (str): plot interpolation.
        save_basename (str): saved png, gif, and video base name.
        fps (int): frame per seconds for gif and video.

    """

    def __init__(
        self,
        im,
        dvf,
        x=-1,
        y=-2,
        z=None,
        c=None,
        hide_axes=False,
        mode=None,
        colormap=None,
        vmin=None,
        vmax=None,
        title="",
        interpolation="nearest",
        save_basename="Figure",
        fps=10,
        quiver_scale=5e-2,
        quiver_step=2,
        quiver_color='tab:orange',
    ):
        if im.ndim < 2:
            raise TypeError(
                "Image dimension must at least be two, got {im_ndim}".format(
                    im_ndim=im.ndim
                )
            )


        import matplotlib.pyplot as plt

        self.axim = None
        self.axquiver = None
        self.im = im
        self.dvf = dvf
        self.quiver_scale = quiver_scale 
        self.quiver_step = quiver_step
        self.quiver_color = quiver_color
        
        self.fig, self.ax = plt.subplots(1, 2)
        self.ax_im = self.ax[0]
        self.ax_quiver = self.ax[1]
        plt.subplots_adjust(wspace=0, hspace=0)
        self.fig.set_facecolor('None')
        self.fig.patch.set_alpha(0.0)
        self.ax_quiver.set_facecolor('None')
        self.ax_quiver.patch.set_alpha(0.0)
        
        self.shape = self.im.shape
        self.ndim = self.im.ndim
        self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.x = x % self.ndim
        self.y = y % self.ndim
        self.z = z % self.ndim if z is not None else None
        self.c = c % self.ndim if c is not None else None
        self.d = max(self.ndim - 3, 0)
        self.hide_axes = hide_axes
        self.show_help = False
        self.title = title
        self.interpolation = interpolation
        self.mode = mode
        self.colormap = colormap
        self.entering_slice = False
        self.vmin = vmin
        self.vmax = vmax
        self.save_basename = save_basename
        self.fps = fps
        self.help_text = None

        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id
        )
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()
        plt.show()

    def key_press(self, event):
        if event.key == "up":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "down":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] - 1) % self.shape[
                    self.d
                ]
            else:
                self.flips[self.d] *= -1

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "left":
            self.d = (self.d - 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "right":
            self.d = (self.d + 1) % self.ndim

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "x" and self.d not in [self.x, self.z, self.c]:
            if self.d == self.y:
                self.x, self.y = self.y, self.x
            else:
                self.x = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "y" and self.d not in [self.y, self.z, self.c]:
            if self.d == self.x:
                self.x, self.y = self.y, self.x
            else:
                self.y = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "z" and self.d not in [self.x, self.y, self.c]:
            if self.d == self.z:
                self.z = None
            else:
                self.z = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif (
            event.key == "c"
            and self.d not in [self.x, self.y, self.z]
            and self.shape[self.d] == 3
        ):
            if self.d == self.c:
                self.c = None
            else:
                self.c = self.d

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "t":
            self.x, self.y = self.y, self.x

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "a":
            self.hide_axes = not self.hide_axes

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "f":
            self.fig.canvas.manager.full_screen_toggle()

        elif event.key == "q":
            self.vmin = None
            self.vmax = None
            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "]":
            width = self.vmax - self.vmin
            self.vmin -= width * 0.1
            self.vmax -= width * 0.1

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "[":
            width = self.vmax - self.vmin
            self.vmin += width * 0.1
            self.vmax += width * 0.1

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "}":
            width = self.vmax - self.vmin
            center = (self.vmax + self.vmin) / 2
            self.vmin = center - width * 1.1 / 2
            self.vmax = center + width * 1.1 / 2

            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "{":
            width = self.vmax - self.vmin
            center = (self.vmax + self.vmin) / 2
            self.vmin = center - width * 0.9 / 2
            self.vmax = center + width * 0.9 / 2

            self.update_image()
            self.fig.canvas.draw()

        elif event.key in ["m", "p", "r", "i", "l"]:
            self.vmin = None
            self.vmax = None
            self.mode = event.key

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "s":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.png"
            )
            self.fig.savefig(
                filename,
                transparent=True,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
            )

        elif event.key == "k":
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i
                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()

        elif event.key == "g":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.gif"
            )
            print(filename)
            temp_basename = uuid.uuid4()

            bbox = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    bbox_inches=bbox,
                    pad_inches=0,
                    dpi=400,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "palettegen",
                    "{} palette.png".format(temp_basename),
                ]
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "image2",
                    "-s",
                    "{}x{}".format(
                        int(bbox.width * self.fig.dpi),
                        int(bbox.height * self.fig.dpi),
                    ),
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-i",
                    "{} palette.png".format(temp_basename),
                    "-lavfi",
                    "paletteuse",
                    filename,
                ]
            )

            os.remove("{} palette.png".format(temp_basename))
            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key == "v":
            filename = self.save_basename + datetime.datetime.now().strftime(
                " %Y-%m-%d at %I.%M.%S %p.mp4"
            )
            temp_basename = uuid.uuid4()

            for i in range(self.shape[self.d]):
                self.slices[self.d] = i

                self.update_axes()
                self.update_image()
                self.fig.canvas.draw()
                self.fig.savefig(
                    "{} {:05d}.png".format(temp_basename, i),
                    format="png",
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-r",
                    str(self.fps),
                    "-i",
                    "{} %05d.png".format(temp_basename),
                    "-vf",
                    "crop=floor(iw/2)*2-10:floor(ih/2)*2-10",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "1",
                    "-vcodec",
                    "libx264",
                    "-preset",
                    "veryslow",
                    filename,
                ]
            )

            for i in range(self.shape[self.d]):
                os.remove("{} {:05d}.png".format(temp_basename, i))

        elif event.key in [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "backspace",
        ] and self.d not in [self.x, self.y, self.z, self.c]:
            if self.entering_slice:
                if event.key == "backspace":
                    if self.entered_slice < 10:
                        self.entering_slice = False
                    else:
                        self.entered_slice //= 10
                else:
                    self.entered_slice = self.entered_slice * 10 + int(
                        event.key
                    )
            elif event.key != "backspace":
                self.entering_slice = True
                self.entered_slice = int(event.key)

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "enter" and self.entering_slice:
            self.entering_slice = False
            if self.entered_slice < self.shape[self.d]:
                self.slices[self.d] = self.entered_slice

                self.update_image()

            self.update_axes()
            self.fig.canvas.draw()

        elif event.key == "h":
            self.show_help = not self.show_help

            self.update_image()
            self.fig.canvas.draw()
        else:
            return

    def update_image(self):
        # Extract slice.
        idx = []
        dvf_idx_horizontal = []
        dvf_idx_vertical = []
        for i in range(self.ndim):
            if i in [self.x, self.y, self.z, self.c]:
                idx.append(slice(None, None, self.flips[i]))
                dvf_idx_horizontal.append(slice(None, None, self.flips[i]))
                dvf_idx_vertical.append(slice(None, None, self.flips[i]))
            else:
                idx.append(self.slices[i])
                dvf_idx_horizontal.append(self.slices[i])
                dvf_idx_vertical.append(self.slices[i])

        if len(self.dvf.shape) <= 4:
            dvf_idx_horizontal.append(self.x)
            dvf_idx_vertical.append(self.y)
        elif len(self.dvf.shape) == 5:
            dvf_idx_horizontal.append(self.x-int(1))
            dvf_idx_vertical.append(self.y-int(1))
        idx = tuple(idx)
        dvf_idx_horizontal = tuple(dvf_idx_horizontal)
        dvf_idx_vertical = tuple(dvf_idx_vertical)
        imv = sp.to_device(self.im[idx])
        dvf_display_x = sp.to_device(self.dvf[dvf_idx_horizontal]*self.flips[self.x])
        dvf_display_y = sp.to_device(self.dvf[dvf_idx_vertical]*self.flips[self.y])
        
        # Transpose to have [z, y, x, c].
        imv_dims = [self.y, self.x]
        if self.z is not None:
            imv_dims = [self.z] + imv_dims

        if self.c is not None:
            imv_dims = imv_dims + [self.c]
        
        imv = np.transpose(imv, np.argsort(np.argsort(imv_dims)))
        dvf_display_x = np.transpose(dvf_display_x, np.argsort(np.argsort(imv_dims)))
        dvf_display_y = np.transpose(dvf_display_y, np.argsort(np.argsort(imv_dims)))
        imv = array_to_image(imv, color=self.c is not None)
        dvf_grid = np.meshgrid(np.arange(imv.shape[1]),np.arange(imv.shape[0]))

        if self.mode is None:
            if np.isrealobj(imv):
                self.mode = "r"
            else:
                self.mode = "m"

        if self.mode == "m":
            imv = np.abs(imv)
        elif self.mode == "p":
            imv = np.angle(imv)
        elif self.mode == "r":
            imv = np.real(imv)
        elif self.mode == "i":
            imv = np.imag(imv)
        elif self.mode == "l":
            imv = np.abs(imv)
            imv = np.log(imv, out=np.ones_like(imv) * -31, where=imv != 0)

        if self.vmin is None:
            self.vmin = imv.min()

        if self.vmax is None:
            self.vmax = imv.max()

        max_y = imv.shape[0]  # Height
        max_x = imv.shape[1]  # Width
        AXIS_ASPECT_SETTINGS = dict(aspect='equal', adjustable='box')
        
        grid_quiver_x = dvf_grid[0][::self.quiver_step,::self.quiver_step]
        grid_quiver_y = dvf_grid[1][::self.quiver_step,::self.quiver_step]
        dvf_quiver_x = dvf_display_x[::self.quiver_step,::self.quiver_step]
        dvf_quiver_y = dvf_display_y[::self.quiver_step,::self.quiver_step]
        # Remove 0 vectors
        magnitude = np.sqrt(dvf_quiver_x**2 + dvf_quiver_y**2)
        mask = magnitude > 1e-6 
        dvf_quiver_x_filtered = dvf_quiver_x[mask]
        dvf_quiver_y_filtered = dvf_quiver_y[mask]
        grid_quiver_x_filtered = grid_quiver_x[mask]
        grid_quiver_y_filtered = grid_quiver_y[mask]

        # Quiver Plot
        if self.axquiver is not None:
            self.axquiver.remove()
            self.axquiver = None
        self.axquiver = self.ax_quiver.quiver(
            grid_quiver_x_filtered,
            grid_quiver_y_filtered,
            dvf_quiver_x_filtered,
            dvf_quiver_y_filtered,
            scale=self.quiver_scale, 
            # units='x', # TODO: new
            color=self.quiver_color,
            width=0.01,      # ADD/CHANGE: Set the arrow shaft width (a good starting value)
            headlength=5,     # CHANGE: Restore or increase head length (default is 5)
            headwidth=3,      # CHANGE: Restore or increase head width (default is 3)
            minlength=1,
            # headlength=0, # remove green dots for 0 MFs
            # headwidth=0, # remove green dots for 0 MFs
            # minlength=0, # remove green dots for 0 MFs
        )
        self.ax_quiver.set_xlim(0, max_x)
        self.ax_quiver.set_ylim(0, max_y)
        self.ax_quiver.set_aspect(**AXIS_ASPECT_SETTINGS)
        self.ax_quiver.autoscale(False)
        
        # Image Plot
        if self.axim is not None:
            self.axim.remove()
            self.axim = None
        if self.colormap is None:
            colormap = "gray"
        else:
            colormap = self.colormap
        self.axim = self.ax_im.imshow(
            imv,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=colormap,
            origin="lower",
            interpolation=self.interpolation,
            aspect='equal',
            extent=[0, max_x, 0, max_y],
        )
        if self.colormap is not None:
            self.colorbar = self.fig.colorbar(self.axim, ax=self.ax_im) 
        # Re-apply AXIS limits and aspect (CRITICAL and always done)
        self.ax_im.set_xlim(0, max_x)
        self.ax_im.set_ylim(0, max_y)
        self.ax_im.set_aspect(**AXIS_ASPECT_SETTINGS)
        self.ax_im.autoscale(False)
            
        if self.help_text is None:
            bbox_props = dict(
                boxstyle="round", pad=1, fc="white", alpha=0.95, lw=0
            )
            self.help_text = self.ax_im.text( # <--- FIX: Use self.ax_im
                imv.shape[1] / 2, # Note: using shape[1] for x-coordinate
                imv.shape[0] / 2, # Note: using shape[0] for y-coordinate
                image_plot_help_str,
                ha="center",
                va="center",
                linespacing=1.5,
                ma="left",
                size=8,
                bbox=bbox_props,
            )
        
        self.help_text.set_visible(self.show_help)

    def update_axes(self):
        if not self.hide_axes:
            caption = "["
            for i in range(self.ndim):
                if i == self.d:
                    caption += "["
                else:
                    caption += " "

                if self.flips[i] == -1 and (
                    i == self.x or i == self.y or i == self.z or i == self.c
                ):
                    caption += "-"

                if i == self.x:
                    caption += "x"
                elif i == self.y:
                    caption += "y"
                elif i == self.z:
                    caption += "z"
                elif i == self.c:
                    caption += "c"
                elif i == self.d and self.entering_slice:
                    caption += str(self.entered_slice) + "_"
                else:
                    caption += str(self.slices[i])

                if i == self.d:
                    caption += "]"
                else:
                    caption += " "
            caption += "]"

            # Set the title and labels for the image axis
            self.ax_im.set_title(caption)
            self.ax_im.xaxis.set_visible(True)
            self.ax_im.yaxis.set_visible(True)
            self.ax_im.title.set_visible(True)
            
            # Optionally set the title and labels for the quiver axis
            self.ax_quiver.set_title("Quiver Plot") 
            self.ax_quiver.xaxis.set_visible(True)
            self.ax_quiver.yaxis.set_visible(False) # To bring them closer
            self.ax_quiver.title.set_visible(True)

            self.fig.suptitle(self.title)
            
        else:
            # Hide everything on both axes
            for ax in self.ax: # Iterate through both axes in self.ax array
                ax.set_title("")
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.title.set_visible(False)
            self.fig.suptitle("")