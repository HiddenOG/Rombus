# Frontend: build the React app, then serve the static files with nginx.
# Works on Railway (which sets $PORT) and with docker compose.

FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# Public settings, baked into the app at build time (Railway passes service variables as build args).
# Only public values belong here: anything in the frontend is visible to every visitor.
ARG VITE_API_URL=http://localhost:8000
ARG VITE_SUPABASE_URL=
ARG VITE_SUPABASE_PUBLISHABLE_KEY=
ENV VITE_API_URL=$VITE_API_URL \
    VITE_SUPABASE_URL=$VITE_SUPABASE_URL \
    VITE_SUPABASE_PUBLISHABLE_KEY=$VITE_SUPABASE_PUBLISHABLE_KEY
RUN npm run build


FROM nginx:1.27-alpine
# nginx fills ${PORT} into templates/*.template at startup
ENV PORT=3000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 3000
