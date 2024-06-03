from django.urls import path
from .views import home

urlpatterns = [
    path('', home.as_view(), name='home'),  # Map the root URL to the 'home' view
]
