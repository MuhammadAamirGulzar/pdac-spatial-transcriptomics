from PIL import Image
import tifffile
import openslide

# Image Filepaths
IU_PDA_T1 = r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_T1-S13_11414_B9.tif"
IU_PDA_T4 = r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_T4-S17_19380_H6.tif"
IU_PDA_T11 = r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_T11-S21_23955_F12.tif"

# Using tifffile (most comprehensive)
with tifffile.TiffFile(IU_PDA_T1) as tif:
    for page in tif.pages:
        print(page.tags)           # all TIFF tags
        tags = page.tags
        # Magnification is often in tag 65000 or similar vendor-specific tags
        for tag in tags.values():
            print(tag.name, tag.value)

with tifffile.TiffFile(IU_PDA_T1) as tif:
    ij_meta = tif.pages[0].tags[50839].value
    print(ij_meta.get('Info'))

slide = openslide.OpenSlide(IU_PDA_T1)
print(slide.properties.get('openslide.objective-power'))
print(slide.properties.get('openslide.mpp-x'))  # µm per pixel, ground truth