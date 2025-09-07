FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
# Install all deps to ensure Next CLI is available at runtime
RUN npm install

COPY . .

RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
