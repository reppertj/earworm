# When running in development, run npm start outside the container and use the react-scripts
# proxy to reach the backend. (https://create-react-app.dev/docs/proxying-api-requests-in-development/)
# This builds a production image using nginx to serve the production-optimized build 
FROM node:12 AS build-stage

WORKDIR /app

COPY ./nginx.conf /nginx.conf

COPY package.json /app/

ENV NODE_PATH=/node_modules
ENV PATH=$PATH:/node_modules/.bin

ARG GENERATE_SOURCEMAP
ARG REACT_APP_SPOTIFY_CLIENT_ID
ARG REACT_APP_REDIRECT_URL

RUN yarn install

COPY ./ /app/

RUN yarn build

FROM nginx:1.18

COPY --from=build-stage /app/build/ /usr/share/nginx/html
COPY --from=build-stage /nginx.conf /etc/nginx/conf.d/default.conf

COPY ./nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found.conf
