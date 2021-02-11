import os
import cv2
from tqdm import tqdm
from bs4 import BeautifulSoup
from cv2 import imread


def annotator(images_path: str, xml_path: str):

    if not images_path.endswith('/'):
        images_path += '/'
    if not xml_path.endswith('/'):
        xml_path += '/'

    kernel_pixel_rate = 5/1000000
    area_pixel_rate = 10000/1000000

    images = sorted(os.listdir(images_path))

    for i in tqdm(range(len(images)), desc='re-annotating images from ' + images_path.split('/')[0]):

        xml_file = open(xml_path + images[i].split('.')[0] + '.xml')
        soup = BeautifulSoup(xml_file.read(), 'xml')
        annotation = soup.annotation

        objs = soup.findAll('object')

        img = imread(images_path + images[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 30)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (int(kernel_pixel_rate * img.shape[0] * img.shape[1]),
                                            int(kernel_pixel_rate * img.shape[0] * img.shape[1])))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            area = cv2.contourArea(c)
            if area > int(area_pixel_rate * img.shape[0] * img.shape[1]):
                x, y, w, h = cv2.boundingRect(c)
                x_center = x + w / 2
                y_center = y + h / 2

                detected = False
                object_type = 'unknown'

                for obj in objs:
                    if int(obj.find('xmin').text) <= x_center <= int(obj.find('xmax').text) and \
                            int(obj.find('ymin').text) <= y_center <= int(obj.find('ymax').text):
                        detected = True
                        object_type = obj.find('name').text
                        break

                if not detected:
                    object_type = 'textbox'

                    new_tag = soup.new_tag('object')
                    annotation.append(new_tag)
                    last_obj = annotation.find_all('object')[-1]
                    new_tag = soup.new_tag('name')
                    new_tag.string = object_type
                    last_obj.append(new_tag)
                    new_tag = soup.new_tag('bndbox')
                    last_obj.append(new_tag)
                    bndbox = last_obj.find('bndbox')
                    new_tag = soup.new_tag('xmin')
                    new_tag.string = str(x)
                    bndbox.append(new_tag)
                    new_tag = soup.new_tag('ymin')
                    new_tag.string = str(y)
                    bndbox.append(new_tag)
                    new_tag = soup.new_tag('xmax')
                    new_tag.string = str(x+w)
                    bndbox.append(new_tag)
                    new_tag = soup.new_tag('ymax')
                    new_tag.string = str(y+h)
                    bndbox.append(new_tag)

        xml_file = open(xml_path + images[i].split('.')[0] + '.xml', "w")
        xml_file.write(soup)
        xml_file.close()


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Prepare dataset for page object detection.')

    parser.add_argument('--images_path', required=True,
                        metavar="path or URL to images",
                        help='Image to apply re-annotation')
    parser.add_argument('--xml_path', required=True,
                        metavar="path or URL to xml files",
                        help='xml files to read/write corresponding annotations')

    args = parser.parse_args()
    assert args.images_path, "Argument --images_path is required"
    assert args.xml_path, "Argument --xml_path is required"

    annotator(images_path=args.images_path, xml_path=args.xml_path)
