# Multi-stage build for frontend and backend integration
FROM node:22-slim AS frontend-builder

# Install pnpm
RUN npm install -g pnpm

# Set working directory for frontend
WORKDIR /frontend

# Copy frontend package files and scripts first
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY .npmrc ./
COPY packages ./packages
COPY apps ./apps
COPY scripts ./scripts
COPY tsconfig.json ./tsconfig.json

# Install frontend dependencies
RUN pnpm install

# Build frontend
RUN pnpm run build

# Production stage
FROM nginx:alpine

# install gettext
RUN apk add --no-cache gettext

# copy nginx.conf.template
COPY nginx.template.conf /etc/nginx/templates/nginx.template.conf

# copy built files from the builder stage
COPY --from=frontend-builder /frontend/apps/web/dist /usr/share/nginx/html

# copy entrypoint.sh
COPY scripts/docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh

ENV BACKEND_HOST=backend
# expose port
EXPOSE 80

# use custom entrypoint script
ENTRYPOINT ["/docker-entrypoint.sh"]

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 