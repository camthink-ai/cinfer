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

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from the builder stage
COPY --from=frontend-builder /frontend/apps/web/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 