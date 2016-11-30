from django.conf import settings
from django.conf.urls import include, url
from django.conf.urls.static import static
from django.contrib import admin
from django.views.generic import TemplateView
from summary.views import parse_api

urlpatterns = [
    url(r'^$', TemplateView.as_view(template_name="index.html"), name='index'),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^summary/', parse_api),
]

# User-uploaded files like profile pics need to be served in development
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
