
from django.shortcuts import render
from django.views.generic import TemplateView
import folium
import ee

from typing import Tuple
import numpy as np
from scipy.ndimage import zoom
from .models import LandCoverModel
import datetime
import logging
import branca

import tensorflow as tf

logger = logging.getLogger(__name__)
ee.Initialize()

# Constants from your preprocessing
PATCH_SIZE = 510
SCALE = 10

def resize_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resizes an image to target shape using bilinear interpolation"""
    zoom_factors = [target_shape[i] / image.shape[i] for i in range(image.ndim)]
    return zoom(image, zoom_factors, order=1)

def get_sentinel2_image(lon: float, lat: float, start_date, end_date, scale=10, max_days=30):

    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
    # In this example, we'll define a point in the DRC.
    POI = ee.Geometry.Point([lon, lat])

    s2 = s2.filterBounds(POI).filterDate(start_date, end_date) \
    .sort('system:time_start', False)
    # Check if any images are available
    if s2.size().getInfo() == 0:
        print("No images found for the specified date and location.")
        return None

    # Grab the first image in the filtered collection. Dynamic World uses a subset
    # of Sentinel-2 bands, so we'll want to select down to just those.
    s2_image = s2.first()
    # Check if an image was found
    if s2_image is None:
        print("No image found for the specified date and location.")
        return None


    # # Compute Spectral Indices using normalizedDifference
    # ndvi = s2_image.normalizedDifference(['B8', 'B4']).rename('NDVI')   # Vegetation
    # ndwi = s2_image.normalizedDifference(['B3', 'B8']).rename('NDWI')   # Water
    # ndbi = s2_image.normalizedDifference(['B11', 'B8']).rename('NDBI')  # Built-up

    # Add the indices to the original image
    # s2_image = s2_image.addBands([ndvi, ndwi, ndbi])

    s2_image = s2_image.select('B2','B3','B4','B5','B6','B7','B8','B11','B12')

    # Resample the data so that the bands all map 1 pixel -> 10m. We'll use B2 (red)
    # for a reference projection.
    s2_image = s2_image.toFloat().resample('bilinear').reproject(
        s2_image.select('B2').projection());

    # Squash the image bands down into an array-per-pixel, and sample out a square
    # from our image centered on the POI. In this example, we'll go out 2km in each
    # direction.
    #
    # This creates an ee.Feature with a property named "array" that we'll grab
    # later.
    s2_image_sample = s2_image.toArray().sampleRectangle(POI.buffer(2000))

    image = np.array(s2_image_sample.getInfo()['properties']['array'])



    # Note this shape isn't exactly 400 a side (2 * 2km of 10m pixels) since the
    # "buffer" we used earlier was in a different (geographic) projection than the
    # pixels.
    #print(image.shape)

    original_image = image

    # Define per-band constants we'll use to squash the Sentinel-2 reflectance range
    # into something on (0, 1). These constants are 30/70 percentiles measured
    # across a diverse set of surface conditions after a log transform.

    NORM_PERCENTILES = np.array([
        [1.7417268007636313, 2.023298706048351],
        [1.7261204997060209, 2.038905204308012],
        [1.6798346251414997, 2.179592821212937],
        [1.7734969472909623, 2.2890068333026603],
        [2.289154079164943, 2.6171674549378166],
        [2.382939712192371, 2.773418590375327],
        [2.3828939530384052, 2.7578332604178284],
        [2.1952484264967844, 2.789092484314204],
        [1.554812948247501, 2.4140534947492487]])

    image = np.log(image * 0.005 + 1)
    image = (image - NORM_PERCENTILES[:, 0]) / NORM_PERCENTILES[:, 1]

    # Get a sigmoid transfer of the re-scaled reflectance values.
    image = np.exp(image * 5 - 1)
    image = image / (image + 1)

    return image

def get_sentinel1_image(lon, lat, start_date, end_date, scale=10, max_days=30):
    lon = float(lon)
    lat = float(lat)

    # In this example, we'll define a point in the DRC.
    POI = ee.Geometry.Point([lon, lat])

    def speckle_filter(image):
        kernel = ee.Kernel.square(7)
        mean = image.reduceNeighborhood(ee.Reducer.mean(), kernel)
        variance = image.reduceNeighborhood(ee.Reducer.variance(), kernel)
        return image.expression(
            'b(0) * (mean / (mean + variance)) + variance / (mean + variance)', {
                'b(0)': image,
                'mean': mean,
                'variance': variance
            })

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(POI) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .sort('system:time_start', False)

    #Function to convert from dB
    def toNatural(img):
      return ee.Image(10.0).pow(img.select(0).divide(10.0))
    #Function to convert to dB
    def toDB(img):
      return ee.Image(img).log10().multiply(10.0)

    import math
    # Implementation by Andreas Vollrath (ESA), inspired by Johannes Reiche (Wageningen)
    def terrainCorrection(image):
      imgGeom = image.geometry()
      srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom) # 30m srtm
      sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

      # Article ( numbers relate to chapters)
      # 2.1.1 Radar geometry
      theta_i = image.select('angle')
      phi_i = ee.Terrain.aspect(theta_i) \
        .reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000) \
        .get('aspect')

      # 2.1.2 Terrain geometry
      alpha_s = ee.Terrain.slope(srtm).select('slope')
      phi_s = ee.Terrain.aspect(srtm).select('aspect')

      # 2.1.3 Model geometry
      # reduce to 3 angle
      phi_r = ee.Image.constant(phi_i).subtract(phi_s)

      # convert all to radians
      phi_rRad = phi_r.multiply(math.pi / 180)
      alpha_sRad = alpha_s.multiply(math.pi / 180)
      theta_iRad = theta_i.multiply(math.pi / 180)
      ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

      # slope steepness in range (eq. 2)
      alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

      # slope steepness in azimuth (eq 3)
      alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

      # local incidence angle (eq. 4)
      theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos()
      theta_liaDeg = theta_lia.multiply(180 / math.pi)
      # 2.2
      # Gamma_nought_flat
      gamma0 = sigma0Pow.divide(theta_iRad.cos())
      gamma0dB = ee.Image.constant(10).multiply(gamma0.log10())
      ratio_1 = gamma0dB.select('VV').subtract(gamma0dB.select('VH'))

      # Volumetric Model
      nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan()
      denominator = (ninetyRad.subtract(theta_iRad)).tan()
      volModel = (nominator.divide(denominator)).abs()

      # apply model
      gamma0_Volume = gamma0.divide(volModel)
      gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10())

      # we add a layover/shadow maskto the original implmentation
      # layover, where slope > radar viewing angle
      alpha_rDeg = alpha_r.multiply(180 / math.pi)
      layover = alpha_rDeg.lt(theta_i)

      # shadow where LIA > 90
      shadow = theta_liaDeg.lt(85)

      # calculate the ratio for RGB vis
      ratio = gamma0_VolumeDB.select('VV').subtract(gamma0_VolumeDB.select('VH'))

      output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad) \
        .addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1)

      return image.addBands(
        output.select(['VV', 'VH'], ['VV', 'VH']),
        None,
        True
      )

    #Applying a Refined Lee Speckle filter as coded in the SNAP 3.0 S1TBX:
    #https:#github.com/senbox-org/s1tbx/blob/master/s1tbx-op-sar-processing/src/main/java/org/esa/s1tbx/sar/gpf/filtering/SpeckleFilters/RefinedLee.java
    #Adapted by Guido Lemoine

    def RefinedLee(img):
      # img must be in natural units, i.e. not in dB!
      # Set up 3x3 kernels

      # convert to natural.. do not apply function on dB!
      myimg = toNatural(img)

      weights3 = ee.List.repeat(ee.List.repeat(1,3),3)
      kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False)

      mean3 = myimg.reduceNeighborhood(ee.Reducer.mean(), kernel3)
      variance3 = myimg.reduceNeighborhood(ee.Reducer.variance(), kernel3)

      # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
      sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]])

      sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False)

      # Calculate mean and variance for the sampled windows and store as 9 bands
      sample_mean = mean3.neighborhoodToBands(sample_kernel)
      sample_var = variance3.neighborhoodToBands(sample_kernel)

      # Determine the 4 gradients for the sampled windows
      gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
      gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
      gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
      gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

      # And find the maximum gradient amongst gradient bands
      max_gradient = gradients.reduce(ee.Reducer.max())

      # Create a mask for band pixels that are the maximum gradient
      gradmask = gradients.eq(max_gradient)

      # duplicate gradmask bands: each gradient represents 2 directions
      gradmask = gradmask.addBands(gradmask)

      # Determine the 8 directions
      directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
      directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
      directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
      directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
      # The next 4 are the not() of the previous 4
      directions = directions.addBands(directions.select(0).Not().multiply(5))
      directions = directions.addBands(directions.select(1).Not().multiply(6))
      directions = directions.addBands(directions.select(2).Not().multiply(7))
      directions = directions.addBands(directions.select(3).Not().multiply(8))

      # Mask all values that are not 1-8
      directions = directions.updateMask(gradmask)

      # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
      directions = directions.reduce(ee.Reducer.sum())

      sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

      # Calculate localNoiseVariance
      sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0])

      # Set up the 7*7 kernels for directional statistics
      rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4))

      diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]])

      rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False)
      diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False)

      # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
      dir_mean = myimg.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
      dir_var = myimg.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

      dir_mean = dir_mean.addBands(myimg.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
      dir_= dir_var.addBands(myimg.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

      # and add the bands for rotated kernels
      for i in range(1, 4, 1):
        dir_mean = dir_mean.addBands(myimg.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_= dir_var.addBands(myimg.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_mean = dir_mean.addBands(myimg.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
        dir_= dir_var.addBands(myimg.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

      # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
      dir_mean = dir_mean.reduce(ee.Reducer.sum())
      dir_= dir_var.reduce(ee.Reducer.sum())

      # A finally generate the filtered value
      varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))

      b = varX.divide(dir_var)

      result = dir_mean.add(b.multiply(myimg.subtract(dir_mean)))
      #return(result)
      return(img.addBands(ee.Image(toDB(result.arrayGet(0))).rename("filter")))

    #print(f"Sentinel-1 date range: {start_date} to {end_date}")
    #print(f"Sentinel-1 collection size: {s1.size().getInfo()}")

    s1_image = s1.first()

    s1_image = terrainCorrection(s1_image)
    s1_image = RefinedLee(s1_image)

    s1_vv = s1_image.select('VV').toFloat()
    s1_vh = s1_image.select('VH').toFloat()

    #s1_vv = speckle_filter(s1_vv)
    #s1_vh = speckle_filter(s1_vh)

    # Sample the image around the POI
    s1_image_sample_vv = s1_vv.toArray().sampleRectangle(region=POI.buffer(2000))
    s1_image_sample_vh = s1_vh.toArray().sampleRectangle(region=POI.buffer(2000))

    image_vv = np.array(s1_image_sample_vv.getInfo()['properties']['array'])
    #print(image_vv)
    image_vh = np.array(s1_image_sample_vh.getInfo()['properties']['array'])

    #print(image_vv.shape)
    #print(image_vh.shape)

    # Normalize the images
    #image_vv = (image_vv - np.min(image_vv)) / (np.max(image_vv) - np.min(image_vv))
    #image_vh = (image_vh - np.min(image_vh)) / (np.max(image_vh) - np.min(image_vh))

    return image_vv, image_vh



# views.py

from django.views.generic import TemplateView
import folium, ee, numpy as np, tensorflow as tf, logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from .models import LandCoverModel

logger = logging.getLogger(__name__)
PATCH_SIZE = 510

def combine_classes(label_array):
    """
    Merge an 11-class map (0–10) into your 8‑class scheme:
      - Scrub (6) → Grass (3)
      - Crops (5) → Trees (2)
      - Built Area (7) & Bare Ground (8) → class 5
      - Snow/Ice (9) → 6
      - Cloud (10) → 7
    """
    mapped = label_array.copy()
    mapped[mapped == 6] = 3
    mapped[mapped == 5] = 2
    mapped[(mapped == 7) | (mapped == 8)] = 5
    mapped[mapped == 9] = 6
    mapped[mapped == 10] = 7
    return mapped

# Final 8 classes + hex colors
CLASSIFICATIONS_FINAL = {
    "No data":                   "000000",
    "Water":                     "419BDF",
    "Crops & Trees":             "397D49",
    "Grass & Scrub":             "88B053",
    "Flooded vegetation":        "7A87C6",
    "Built Area & Bare Ground":  "C4281B",
    "Snow/Ice":                  "B39FE1",
    "Cloud":                     "FFFFFF"
}

class home(TemplateView):
    template_name = 'gee/index.html'

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)

        # 1) Read inputs
        lat = float(self.request.GET.get('lat', 22.139527120131657))
        lon = float(self.request.GET.get('lon', 88.85593401040623))
        sd  = self.request.GET.get('start_date', '2025-04-16')
        ed  = self.request.GET.get('end_date',   '2025-04-23')

        # 2) Build Folium map
        m = folium.Map(
            location=[lat, lon],
            zoom_start=14,
            tiles='Esri.WorldImagery',
            attr='Esri'
        )
        folium.Marker([lat, lon]).add_to(m)
        folium.ClickForMarker().add_to(m)
        map_html = m.get_root().render()

        try:
            # 3) Fetch & preprocess imagery
            s2 = get_sentinel2_image(lon, lat, sd, ed)
            s1_vv, s1_vh = get_sentinel1_image(lon, lat, sd, ed)
            for arr,name in [(s2,'Sentinel‑2'), (s1_vv,'VV'), (s1_vh,'VH')]:
                if arr is None or np.isnan(arr).any() or np.isinf(arr).any():
                    raise ValueError(f"Invalid {name} data")

            s2r = resize_image(s2,   (PATCH_SIZE, PATCH_SIZE, s2.shape[-1]))
            vv  = resize_image(np.squeeze(s1_vv), (PATCH_SIZE, PATCH_SIZE))
            vh  = resize_image(np.squeeze(s1_vh), (PATCH_SIZE, PATCH_SIZE))

            # Vegetation indices
            ndvi = np.clip(((s2r[:,:,6] - s2r[:,:,2]) /
                            (s2r[:,:,6] + s2r[:,:,2] + 1e-10) + 1)/2, 0,1)
            ndwi = np.clip(((s2r[:,:,6] - s2r[:,:,7]) /
                            (s2r[:,:,6] + s2r[:,:,7] + 1e-10) + 1)/2, 0,1)
            evi  = np.clip(2.5 * ((s2r[:,:,6] - s2r[:,:,2]) /
                            (s2r[:,:,6] + 6*s2r[:,:,2] -
                             7.5*s2r[:,:,0] + 1)), 0,1)
            arvi = np.clip(((s2r[:,:,6] - 2*s2r[:,:,2] + s2r[:,:,0]) /
                            (s2r[:,:,6] + 2*s2r[:,:,2] + s2r[:,:,0] + 1e-10) + 1)/2, 0,1)

            # === Dynamic World (10-class) inference ===
            dw_model = tf.saved_model.load(
                '/Users/shashankdutt/Downloads/dynamicworld-1.0.0/model/forward'
            )
            x_dw     = tf.expand_dims(tf.cast(s2r, tf.float32), 0)
            dw_logits= dw_model(x_dw)  # (1,H,W,10)
            dw_raw   = np.argmax(tf.nn.softmax(dw_logits)[0].numpy(), axis=-1)
            dw_full  = dw_raw + 1      # shift 0→1, …, 9→10
            dw_lbl8  = combine_classes(dw_full)

            # === Your SavedModel (11-class) inference ===
            sm_model = tf.saved_model.load('/Users/shashankdutt/Downloads/saved_model 2/my_model')
            infer    = sm_model.signatures['serving_default']
            BANDS    = [
              'B2','B3','B4','B5','B6','B7','B8','B11','B12',
              'VV','VH','ndvi','ndwi','evi','arvi'
            ]
            band_data = {
              'B2': s2r[:,:,0], 'B3': s2r[:,:,1], 'B4': s2r[:,:,2],
              'B5': s2r[:,:,3], 'B6': s2r[:,:,4], 'B7': s2r[:,:,5],
              'B8': s2r[:,:,6], 'B11':s2r[:,:,7],'B12':s2r[:,:,8],
              'VV': vv, 'VH': vh,
              'ndvi': ndvi,'ndwi': ndwi,'evi': evi,'arvi': arvi
            }
            keys = list(infer.structured_input_signature[1].keys())
            inp_sm = {
              keys[i]: tf.cast(
                tf.expand_dims(tf.expand_dims(band_data[b], 0), -1),
                tf.float32
              )
              for i,b in enumerate(BANDS)
            }
            sm_logits = infer(**inp_sm)['output_0']  # (1,H,W,11)
            sm_lbl11  = np.argmax(tf.nn.softmax(sm_logits)[0].numpy(), axis=-1)
            sm_lbl8   = combine_classes(sm_lbl11)

            # === Color mapping & plotting ===
            CLASS_COL = np.array(
              [[int(h[i:i+2],16) for i in (0,2,4)]
               for h in CLASSIFICATIONS_FINAL.values()]
            ) / 255.0

            dw_rgb = CLASS_COL[dw_lbl8]
            sm_rgb = CLASS_COL[sm_lbl8]

            fig, ax = plt.subplots(1,3, figsize=(18,6))
            ax[0].imshow(s2r[:,:, [2,1,0]])
            ax[0].axis('off'); ax[0].set_title('RGB Image')
            ax[1].imshow(dw_rgb)
            ax[1].axis('off'); ax[1].set_title('Dynamic World')
            ax[2].imshow(sm_rgb)
            ax[2].axis('off'); ax[2].set_title('Saved Model')

            # --- Alternate plotting (commented out) ---
            # boundaries = np.arange(len(CLASSIFICATIONS_FINAL)+1)
            # cmap = matplotlib.colors.ListedColormap(CLASS_COL)
            # norm = BoundaryNorm(boundaries, cmap.N)
            # fig2, ax2 = plt.subplots(figsize=(6,6))
            # ax2.imshow(sm_lbl8, cmap=cmap, norm=norm)
            # ax2.set_title('SavedModel Only'); ax2.axis('off')

            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            ctx.update({
                'map_html':         map_html,
                'prediction_plot':  plot_b64,
                'prediction_status':'Prediction successful'
            })

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            ctx.update({
                'map_html':         map_html,
                'prediction_status':f'Error: {e}'
            })

        # 7) Legend & form defaults
        ctx.update({
            'lat':            lat,
            'lon':            lon,
            'start_date':     sd,
            'end_date':       ed,
            'classifications': CLASSIFICATIONS_FINAL
        })
        return ctx






















'''
class home(TemplateView):
    template_name = 'gee/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get user inputs
        lat = float(self.request.GET.get('lat', 22.5))
        lon = float(self.request.GET.get('lon', 87.3))
        start_date = self.request.GET.get('start_date', '2024-01-01')
        end_date = self.request.GET.get('end_date', '2024-01-10')

        # Create Folium map
        # figure = folium.Figure()
        m1 = folium.Map(
            location=[lat, lon],
            zoom_start=14,
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite'
        )
        # m.add_to(figure)

        fig = None
        model_instance = LandCoverModel.get_instance()

                # Visualization parameters.
        class_viz = {
            'min': 0,
            'max': 7,
            'palette': ['000000', '419BDF', '397D49', '88B053',
                        '7A87C6', 'C4281B', 'B39FE1', 'FFFFFF']
        }

        try:
            # Get satellite images
            s2_image = get_sentinel2_image(lon, lat, start_date, end_date)
            s1_vv, s1_vh = get_sentinel1_image(lon, lat, start_date, end_date)

            if s2_image is None or s1_vv is None or s1_vh is None:
                raise ValueError("Could not retrieve all satellite data")

            # Resize images to match the expected dimensions.
            s2_resized = resize_image(s2_image, (PATCH_SIZE, PATCH_SIZE, s2_image.shape[-1]))
            s1_vv_resized = resize_image(np.squeeze(s1_vv), (PATCH_SIZE, PATCH_SIZE))
            s1_vh_resized = resize_image(np.squeeze(s1_vh), (PATCH_SIZE, PATCH_SIZE))

            # Calculate spectral indices.
            nir = s2_resized[:, :, 6]  # B8
            red = s2_resized[:, :, 2]  # B4
            blue = s2_resized[:, :, 0] # B2
            swir1 = s2_resized[:, :, 7]  # B11

            ndvi = np.divide(nir - red, nir + red + 1e-10)
            ndwi = np.divide(nir - swir1, nir + swir1 + 1e-10)
            evi = 2.5 * np.divide(nir - red, nir + 6 * red - 7.5 * blue + 1)
            arvi = np.divide(nir - 2 * red + blue, nir + 2 * red + blue + 1e-10)

            # Normalize indices.
            ndvi = np.clip((ndvi + 1) / 2, 0, 1)
            ndwi = np.clip((ndwi + 1) / 2, 0, 1)
            evi = np.clip(evi, 0, 1)
            arvi = np.clip((arvi + 1) / 2, 0, 1)

            # Concatenate inputs along the channel axis.
            # The order here must match the order expected when the model was exported.
            model_input = np.concatenate([
                s2_resized,
                s1_vv_resized[..., np.newaxis],
                s1_vh_resized[..., np.newaxis],
                ndvi[..., np.newaxis],
                ndwi[..., np.newaxis],
                evi[..., np.newaxis],
                arvi[..., np.newaxis]
            ], axis=-1)  # Expected shape: (510, 510, 15)

            # Convert to tensor and add the batch dimension.
            model_input_tf = tf.convert_to_tensor(model_input, dtype=tf.float32)
            model_input_tf = tf.expand_dims(model_input_tf, axis=0)  # Now shape: (1, 510, 510, 15)

            # Get model instance and run prediction.
            model_instance = LandCoverModel.get_instance()
            class_prediction, _ = model_instance.predict(model_input_tf)

            # Convert prediction (2D numpy array) to an EE image.
            arr_img = ee.Image(ee.Array(class_prediction.tolist()))
            proj   = arr_img.arrayProject([0])
            flat   = proj.arrayFlatten([['classification']])
            prediction_image = flat.toByte() \
                .setDefaultProjection(crs='EPSG:4326', scale=SCALE)
            
            
            # Use the right variable name, prediction_image
            pid = prediction_image.getMapId(class_viz)

            # Build your second blank map
            m2 = folium.Map(location=[lat, lon],
                            zoom_start=14,
                            tiles=None)

            folium.TileLayer(
                tiles=pid['tile_fetcher'].url_format,
                attr='Model Prediction',
                name='Land Cover',
                overlay=True,
                control=True
            ).add_to(m2)
            folium.LayerControl().add_to(m2)

            
            fig = branca.element.Figure()
            ax1 = fig.add_subplot(2, 1, 1)  # row=1
            ax2 = fig.add_subplot(2, 1, 2)  # row=2
            ax1.add_child(m1)
            ax2.add_child(m2)
            fig.render()



            context['prediction_status'] = 'Prediction successful'

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            context['prediction_status'] = f'Error: {str(e)}'

        # # Add base layer and controls.
        # folium.TileLayer(
        #     tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        #     attr='Google Maps',
        #     name='Street Map',
        #     overlay=True
        # ).add_to(m)
        # m.add_child(folium.LayerControl())

        if fig:
            map_html = fig.get_root().render()
        else:
            map_html = "<p>Unable to render map.</p>"

        # figure.render()

        context.update({
            'map_html': map_html,
            'lat': lat,
            'lon': lon,
            'start_date': start_date,
            'end_date': end_date,
            'classifications': model_instance.classifications
        })

        return context

'''




''' # First basic stuff - WORKS 

#home
class home(TemplateView):
    template_name = 'index.html'

    # Define a method for displaying Earth Engine image tiles on a folium map.
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        start_date = self.request.GET.get('start_date', '2024-01-01')
        end_date = self.request.GET.get('end_date', '2024-05-30')

        figure = folium.Figure()
        

        #create Folium Object
        m = folium.Map(
            location=[23, 88],
            zoom_start=8
        )

        #add map to figure
        m.add_to(figure)

        #folium imports
        # Add custom base maps to folium
        basemaps = {
            'Google Maps': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Maps',
                overlay = True,
                control = True
            ),
            'Google Satellite': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True
            ),
            'Google Terrain': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Terrain',
                overlay = True,
                control = True
            ),
            'Google Satellite Hybrid': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True
            ),
            'Esri Satellite': folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                overlay = True,
                control = True
            )
        }

        # Add custom basemaps
        basemaps['Google Maps'].add_to(m)
        basemaps['Google Satellite Hybrid'].add_to(m)


        S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start_date, end_date) \
            .filterBounds(ee.Geometry.Polygon(
                [[[88.5034748557929, 22.401058301237384],
                [88.5034748557929, 21.547808553681804],
                [89.17638745344915, 21.547808553681804],
                [89.17638745344915, 22.401058301237384]]])) \
        
        vizParams = {'bands': ['B8', 'B4', 'B3'], 'min': 600, 'max': 1000, 'opacity': 1.0, 'gamma': 1.0}

         #add the map to the the folium map
        map_id_dict = ee.Image(S2.median()).getMapId(vizParams)
       
        #GEE raster data to TileLayer
        #GEE raster data to TileLayer
        folium.raster_layers.TileLayer(
                    tiles = map_id_dict['tile_fetcher'].url_format,
                    attr = 'Google Earth Engine',
                    name = 'Solar annual radiation',
                    overlay = True,
                    control = True
                    ).add_to(m)

        # Load Google Research Open Buildings data
        buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")

        # Add buildings data to the map
        buildings_map_id = buildings.style(color='blue').getMapId()

        folium.raster_layers.TileLayer(
            tiles=buildings_map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name='Open Buildings',
            overlay=True,
            control=True
        ).add_to(m)

        #add Layer control
        m.add_child(folium.LayerControl())
       
        #figure 

        figure.render()
        context['map'] = figure
        context['start_date'] = start_date
        context['end_date'] = end_date 
        #return map
        return context

        

'''

'''
        #select the Dataset Here's used the MODIS data
        dataset = ee.ImageCollection('MODIS/006/MOD13Q1') \
                  .filter(ee.Filter.date('2019-07-01', '2019-11-30')) \
                .filterBounds(ee.Geometry.Polygon(
                [[[88.5034748557929, 22.401058301237384],
                [88.5034748557929, 21.547808553681804],
                [89.17638745344915, 21.547808553681804],
                [89.17638745344915, 22.401058301237384]]])) \
                .first()
        
        modisndvi = dataset.select('NDVI')

        #Styling 
        vis_paramsNDVI = {
            'min': 0,
            'max': 9000,
            'palette': [ 'FE8374', 'C0E5DE', '3A837C','034B48',]}

        
        #add the map to the the folium map
        map_id_dict = ee.Image(modisndvi).getMapId(vis_paramsNDVI)
       
        #GEE raster data to TileLayer
        folium.raster_layers.TileLayer(
                    tiles = map_id_dict['tile_fetcher'].url_format,
                    attr = 'Google Earth Engine',
                    name = 'NDVI',
                    overlay = True,
                    control = True
                    ).add_to(m)
'''
                    
