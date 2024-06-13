import ee
from datetime import datetime, timedelta, timezone
from .models import SatelliteImage

def fetch_latest_image():
    # Initialize the Earth Engine library (Make sure authentication is handled separately)
    ee.Initialize()

    collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(ee.Geometry.Polygon(
        [[[88.5034748557929, 22.401058301237384],
          [88.5034748557929, 21.547808553681804],
          [89.17638745344915, 21.547808553681804],
          [89.17638745344915, 22.401058301237384]]]))

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=500)

    sorted_collection = collection.filterDate(start_date, end_date).sort('system:time_start', False)

    latest_image = sorted_collection.first()

    latest_image_info = latest_image.getInfo()  # Get full metadata
    latest_date = latest_image_info['properties']['system:time_start'] / 1000  # Convert milliseconds to seconds
    latest_date = datetime.fromtimestamp(latest_date, tz=timezone.utc).isoformat()  # Convert to ISO format with UTC

    # Check if the latest image is already in the database
    if not SatelliteImage.objects.filter(date=latest_date).exists():
        image_id = latest_image_info['id']  # Image ID from the metadata
        metadata = latest_image_info

        # Save the new image metadata to the database
        SatelliteImage.objects.create(image_id=image_id, date=latest_date, metadata=metadata)
        print(f'New image added: {image_id} at {latest_date}')
    else:
        print('No new images found')
