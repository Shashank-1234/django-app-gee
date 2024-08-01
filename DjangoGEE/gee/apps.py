from django.apps import AppConfig
import ee

class MyappConfig(AppConfig):
    name = 'gee'

    def ready(self):
        ee.Authenticate()
        ee.Initialize(project='ee-sagirajushashank')
