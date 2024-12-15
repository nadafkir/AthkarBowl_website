from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),  
    path('athkar-bowl/', views.athkar_bowl, name="athkar-bowl"),
    path('athkar-recommendation/', views.athkar_recommendation, name='recommend_athkar'),
    path('athkar/', views.athkar, name='athkar'),
    path('about-us/', views.about_us, name='about-us'), 
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)