# # Use official Node.js image for building
# FROM node:18 AS build

# # Set working directory
# WORKDIR /app

# # Copy package.json and package-lock.json
# COPY package.json package-lock.json ./

# # Install dependencies
# RUN npm install

# # Copy frontend code
# COPY src/ ./src/
# COPY index.html .
# COPY tsconfig.json .
# COPY tsconfig.node.json .
# COPY vite.config.ts .  

# # Build the React app (Vite)
# RUN npm run build

# # Use nginx to serve the built app
# FROM nginx:alpine

# # Copy built files from the build stage
# COPY --from=build /app/dist /usr/share/nginx/html

# # Copy custom nginx configuration
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# # Expose Vite port
# EXPOSE 5173

# # Start nginx
# CMD ["nginx", "-g", "daemon off;"]






# ---------- Build Stage ----------
FROM node:18 AS build

# Set working directory
WORKDIR /app

# Copy dependency files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all necessary files for the build
COPY . .

# Build the Vite app
RUN npm run build


# ---------- Production Stage ----------
FROM nginx:alpine

# Copy built files from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom nginx configuration (replace default)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose the default Nginx port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
