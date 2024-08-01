from django.shortcuts import render

# Create your views here.
# generic base view
from django.views.generic import TemplateView 


#folium
import folium


#gee
import ee

ee.Initialize()




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
       
        

       