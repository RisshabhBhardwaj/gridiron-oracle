FROM node:20-alpine AS build

WORKDIR /app

# Install deps first (better layer caching)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build production assets
COPY frontend .
RUN npm run build          # outputs to /app/dist

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
