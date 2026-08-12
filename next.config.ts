import type { NextConfig } from 'next';

const repoName = 'my-site';
const isGithubPages = process.env.GITHUB_PAGES === 'true';
const basePath = isGithubPages ? `/${repoName}` : '';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  agentRules: false,
  output: 'export',
  images: {
    unoptimized: true,
  },
  basePath,
  assetPrefix: isGithubPages ? `${basePath}/` : undefined,
  trailingSlash: true,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
};

export default nextConfig;
