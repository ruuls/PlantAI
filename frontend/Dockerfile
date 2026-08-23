# Optional: serve the console from nginx and point it at a separate API host.
# The backend image already serves these files, so this is only needed when the
# two are deployed apart.
FROM nginx:1.27-alpine

COPY frontend/ /usr/share/nginx/html/
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
