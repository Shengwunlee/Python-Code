# 20190402 Analytical spectral image use iPhone camera.
#!/usr/bin/python
# -*- coding utf-8 -*-

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from fractions import Fraction
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFile, ImageFont
from matplotlib import pyplot as plt
import random

# find aperture on right hand side of image along middle line
def find_aperture(Phone_pixels, pic_width: int, pic_height: int) -> object:
    middle_x = int(pic_width / 2) #600
    middle_y = int(pic_height / 2) #600
    aperture_brightest = 0
    aperture_x = 0
    for x in range(middle_x, pic_width, 1): # x==middle_x==600, pic_width==1200
        r, g, b = Phone_pixels[x, middle_y] #600,601,602,...,1200
        brightness = r + g + b
        if brightness > aperture_brightest:
            aperture_brightest = brightness
            aperture_x = x #1124

    aperture_threshold = aperture_brightest * 0.9 #Aperture up and down size(1.0/0.9/0,7/0.5)

    aperture_x1 = aperture_x #1124
    for x in range(aperture_x, middle_x, -1):
        r, g, b = Phone_pixels[x, middle_y]
        brightness = r + g + b
        if brightness < aperture_threshold:
            aperture_x1 = x
            break

    aperture_x2 = aperture_x
    for x in range(aperture_x, pic_width, 1):
        r, g, b = Phone_pixels[x, middle_y]
        brightness = r + g + b
        if brightness < aperture_threshold:
            aperture_x2 = x
            break

    aperture_x = (aperture_x1 + aperture_x2) / 2
    # print(aperture_x)

    spectrum_threshold_duration = 64 #64
    aperture_y_bounds = get_spectrum_y_bound(Phone_pixels, aperture_x, middle_y, aperture_threshold, spectrum_threshold_duration)
    aperture_y = (aperture_y_bounds[0] + aperture_y_bounds[1]) / 2
    aperture_height = (aperture_y_bounds[1] - aperture_y_bounds[0])
    # Solution of height err.
    return {'x': aperture_x, 'y': aperture_y, 'h': aperture_height, 'b': aperture_brightest}

# scan a column to determine top and bottom of area of lightness
def get_spectrum_y_bound(pix, x, middle_y, spectrum_threshold, spectrum_threshold_duration):
    c = 0
    spectrum_top = middle_y
    for y in range(middle_y, 0, -1):
        r, g, b = pix[x, y]
        brightness = r + g + b
        if brightness < spectrum_threshold: # aperture_threshold
            c = c + 1
            if c > spectrum_threshold_duration:
                break
        else:
            spectrum_top = y
            c = 0

    c = 0
    spectrum_bottom = middle_y
    for y in range(middle_y, middle_y * 2, 1):
        r, g, b = pix[x, y]
        brightness = r + g + b
        if brightness < spectrum_threshold:
            c = c + 1
            if c > spectrum_threshold_duration:
                break
        else:
            spectrum_bottom = y

            c = 0
    return spectrum_top, spectrum_bottom

# draw aperture onto image
def draw_aperture(aperture, draw):
    fill_color = "Green"
    draw.line((aperture['x'], aperture['y'] - aperture['h'] / 2,
        aperture['x'], aperture['y'] + aperture['h'] / 2),
               width=4,fill=fill_color)

# draw scan line
def draw_scan_line(aperture, draw, spectrum_angle,pic_width: int, pic_height: int):
    fill_color = "#888"
    xd = aperture['x']
    h = aperture['h']/2 #2.3
    y_Spec = (math.tan(spectrum_angle) * aperture['x'] + aperture['y'])
    # draw.line((0, y_Spec - h, aperture['x'], aperture['y'] - h), width=3, fill=fill_color) #top
    # print(xd,(y_Spec + h),(aperture['y'] + h))
    draw.line((0, y_Spec + h, xd, aperture['y'] + h), width=5, fill=fill_color) #down

def draw_graph(draw, Phone_pixels, aperture: object, spectrum_angle, wavelength_factor):
    aperture_height = aperture['h'] / 2 #2.3
    # print(aperture['h'],aperture_height)
    last_graph_y = 0
    max_result = 0
    results = OrderedDict()
    for Spectrum_x in range(0, int(aperture['x']), 1): # * 7 / 8
    #hint_1
        wavelength = round((((aperture['x'] - Spectrum_x) * wavelength_factor)+70),0) #47/63/53/90/100,70
        if 720 <= wavelength or wavelength <= 380: # 900,380,721,379|755,71
        # Confirmed in the spectral range
        	continue

        # general efficiency curve of 1000/mm grating
        # hint_2_norm.
        wavalength_normalize = round(((720 - (wavelength - 380)) / 720),5) #800/250,750/250,680/400,+/-1
        # print(normalize)
        if wavalength_normalize < 0.3:
        	wavalength_normalize = 0.3

        # notch near yellow maybe caused by camera sensitivity
        mid = 540
        width = 10
        if (mid - width) < wavelength < (mid + width):
        	d = (width - abs(wavelength - mid)) / width
        	wavalength_normalize = wavalength_normalize * (1 - d * 0.12) #0.12

        # # up notch near 590
        # mid = 588
        # width = 10
        # if (mid - width) < wavelength < (mid + width):
        # 	d = (width - abs(wavelength - mid)) / width
        # 	wavalength_normalize = wavalength_normalize * (1 + d * 0.1) #0.1/0.2

        #hint_3
        y_Spec = (math.tan(spectrum_angle) * (aperture['x'] - Spectrum_x) + aperture['y'])
        amplitude = 0
        ac = 0.0 #0.0
        # print(aperture_height,y_Spec)
        for y in range(int(y_Spec - aperture_height), int(y_Spec + aperture_height), 1):
        	r, g, b = Phone_pixels[Spectrum_x, y]
        	q = (r + b + g) * 0.5 #0.5/0.9
        	if y < (y_Spec - aperture_height) or y > (y_Spec + aperture_height): #+2/+2
        		q = q * 0.5 # *0.5/0.9
        	amplitude = (amplitude + q)
        	ac = ac + 1.5 #1.5,1.9
        amplitude = (amplitude / ac) / wavalength_normalize

        results[str(wavelength)] = int(amplitude) #str
        if amplitude > max_result:
            max_result = amplitude
        graph_y = (amplitude/200) * aperture_height #100,180
        draw.line((Spectrum_x - 1, y_Spec + aperture_height - last_graph_y, Spectrum_x, y_Spec + aperture_height - graph_y), width=3, fill="yellow")
        last_graph_y = graph_y
    draw_ticks_and_frequencies(draw, aperture, spectrum_angle, wavelength_factor)
    return results, max_result

def draw_ticks_and_frequencies(draw, aperture, spectrum_angle, wavelength_factor):
    aperture_height = aperture['h'] / 2 #2.28
    for wl in range(380, 721, 10): #901
        # >>>>>>>>>> "+50 / +110 / +90 " is extra
        x = (aperture['x'] + 115) - (wl / wavelength_factor) #115,130
        y_Spec = math.tan(spectrum_angle) * (aperture['x'] - x) + (aperture['y'])
        draw.line((x, y_Spec + aperture_height + 4, x, y_Spec + aperture_height - 4), width=5, fill="#fff")

    for wl in range(380, 721, 40): #901
        # >>>>>>>>>> "+50 / +110 / +90 " is extra
        x = ((aperture['x'] + 115) - (wl / wavelength_factor)) #115,130
        y_Spec = (math.tan(spectrum_angle) * (aperture['x'] - x) + (aperture['y']))
        draw.line((x, y_Spec + aperture_height + 10, x, y_Spec + aperture_height - 10), width=5, fill="#fff")
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
        draw.text((x-10 , y_Spec + aperture_height + 20), str(wl), font=font, fill="#fff")
        #30

def save_image_with_overlay(im, name):
    output_filename = name + "_out.jpg"
    ImageFile.MAXBLOCK = 2 ** 20
    im.save(output_filename, "JPEG", quality=100, optimize=True, progressive=True)

def normalize_results(results, max_result):
    for wavelength in results:
    	results[wavelength] = (results[wavelength] / max_result)
    return results

def export_csv(name, normalized_results):
    csv_filename = name + ".csv"
    csv = open(csv_filename, 'w')
    csv.write("Wavelength,Value\n")
    for wavelength in normalized_results:
        csv.write(wavelength)
        csv.write(",")
        Normalized_Results = (normalized_results[wavelength]) * 1000
        csv.write("{:0.2f}".format(Normalized_Results))
        csv.write("\n")
    csv.close()

#hint_4
def export_diagram(name, normalized_results):
    antialias = 10
    w = 400 * antialias # Weight 2000
    h2 = 200 * antialias # Height 1200

    h = h2 - 20 * antialias #20/10 image hight 2320 Line Height
    sd = Image.new('RGB', (w, h2), (255, 255, 255)) #New image to Draw
    draw = ImageDraw.Draw(sd)

    w1 = 370.0
    w2 = 721.0 #780
    f1 = round((0.9 / w1),5) #0.9/1.0
    f2 = round((1.0 / w2),5) #1.0
    # print(f1,f2)
    for x in range(0, w, 1):
        # Iterate across frequencies, not wavelengths
        lambda2 = round((1.0 / (f1 - (float(x) / float(w) * round((f1 - f2),5)))),0) #1.0
        # print(lambda2)
        c = wavelength_to_color(lambda2)
        draw.line((x, 0, x, h), fill=c)

    pl = [(w, 0), (w, h)]
    for wavelength in normalized_results:
        wl = float(wavelength)
        x = int((wl - w1) / (w2 - w1) * w)
        value = int((1 - normalized_results[wavelength]/2) * h)
        pl.append((int(x), value))
    pl.append((0, h))
    pl.append((0, 0))
    draw.polygon(pl, fill="#fff")
    draw.polygon(pl)

    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 12 * antialias)
    draw.line((1, h, w, h), fill="#000", width=antialias)

    for wl in range(340, 780, 10):
        x = int((float(wl) - w1) / (w2 - w1+100) * w * 1.18) #50,1.3 / 98,1.18
        draw.line((x, h, x, h + 3 * antialias), fill="#000", width=antialias)

    for wl in range(340, 780, 40):
        x = int((float(wl) - w1) / (w2 - w1+100) * w * 1.18) #50,1.3 / 98,1.18
        draw.line((x, h, x, h + 5 * antialias), fill="#000", width=antialias)
        wls = str(wl)
        tx = draw.textsize(wls, font=font)
        draw.text((x - tx[0] / 2, h + 5 * antialias), wls, font=font, fill="#000")

    # save chart
    sd = sd.resize((int(w / antialias), int(h / antialias)), Image.ANTIALIAS)
    output_filename = name + "_chart.png"
    sd.save(output_filename, "PNG", quality=100, optimize=True, progressive=True)

# return an RGB visual representation of wavelength for chart
# Based on: http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm
# The foregoing is based on: http://www.midnightkite.com/color.html
#                vio  blu  cyn  gre  yel  org  red
# thresholds = [ 380, 440, 490, 510, 580, 645, 780 ] #Web camera
#              [ 420, 440, 500, 520, 540, 560, 780 ] #iPhone Camera
#              [ 420, 495, 500, 520, 540, 560, 780 ] #Samsung Camera
#              [ 420, 440, 500, 520, 540, 560, 780 ] #0513
#              380, 448, 490, 510, 540 ,560 ,780
#              380, 465, 480, 490, 540 ,560 ,780
def wavelength_to_color(lambda2):
    factor = 0.0
    color = [0, 0, 0] #R,G,B
    thresholds = [ 390, 460, 490, 510, 540 ,560 ,780 ]
    for i in range(0, 6, 1): #Six in the array
        t1 = thresholds[i]
        t2 = thresholds[i + 1]
        if lambda2 < t1 or lambda2 >= t2: #i = 1~4
            # print(lambda2,t1,t2)
            continue
        if i % 2 != 0: #i = 1,3,5
            tmp = t1
            t1 = t2
            t2 = tmp
        if i < 5: #i = 1~4
            color[i % 3] = (lambda2 - t2) / (t1 - t2)
        color[2 - int(i / 2)] = 1.0 #1.0
        factor = 1.0 #1.0
        break

    # Let the intensity fall off near the vision limits
    if 370 <= lambda2 < 420: #420/440
        factor = 0.2 + 0.8 * (lambda2 - 380) / (420 - 380) #0.2/0.8
    elif 600 <= lambda2 < 780: #580/600/780
        factor = 0.2 + 0.8 * (780 - lambda2) / (780 - 600) #0.2/0.8
    return int(255 * color[0] * factor), int(255 * color[1] * factor), int(255 * color[2] * factor)


def main():
    # 1. Take picture
    name = sys.argv[1]
    raw_filename = (name+".JPG")

    # 2. Get picture's aperture
    image = Image.open(raw_filename)
    img = image.resize((1200,1200),Image.ANTIALIAS) #img,1200
    # im = img.rotate(180)
    im = img.rotate(360)
    plt.show(im)
    print("locating aperture")
    Phone_pixels = im.load()

    aperture = find_aperture(Phone_pixels, im.size[0], im.size[1])

    # 3. Draw aperture and scan line
    spectrum_angle = -0.07 #-0.01, 0.04, -0.025, -0.07/0.07
    draw = ImageDraw.Draw(im)
    draw_aperture(aperture, draw)
    draw_scan_line(aperture, draw, spectrum_angle, im.size[0], im.size[1])

    # 4. Draw graph on picture
    print("analysing image")
    wavelength_factor = 0.61  # 1000/mm,0.892,0.61548,0.61
    # wavelength_factor = 0.892
    results, max_result = draw_graph(draw, Phone_pixels, aperture, spectrum_angle, wavelength_factor)

    # 6. Save picture with overlay
    save_image_with_overlay(im, name)

    # 7. Normalize results for export
    print("exporting CSV")
    normalized_results = normalize_results(results, max_result)

    # 8. Save csv of results
    export_csv(name, normalized_results)

    # 9. Generate spectrum diagram
    print("generating chart")
    export_diagram(name, normalized_results)


main()
