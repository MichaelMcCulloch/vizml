# syntax=docker/dockerfile:1.7
FROM oven/bun:1.4.4 AS build
WORKDIR /app
COPY package.json bun.lock ./
COPY patches ./patches
RUN --mount=type=cache,target=/root/.bun/install/cache bun install --frozen-lockfile
COPY tsconfig.json index.html ./
COPY src ./src
COPY scripts ./scripts
COPY public ./public
COPY templates ./templates
RUN bun run build

FROM oven/bun:1.4.4-slim
WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0 PORT=5173
COPY --from=build /app/dist ./dist
COPY --from=build /app/scripts/server.ts ./scripts/server.ts
COPY --from=build /app/index.html ./index.html
COPY --from=build /app/src ./src
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json
EXPOSE 5173
CMD ["bun", "scripts/server.ts", "--production"]
