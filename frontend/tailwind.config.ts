import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#111318',
        panel: '#f7f4ed',
        paper: '#fffdf8',
        moss: '#315f4b',
        brass: '#b07738',
        marine: '#2c6f87',
        berry: '#8b3f5f',
      },
      boxShadow: {
        soft: '0 18px 50px rgba(22, 24, 29, 0.12)',
      },
    },
  },
  plugins: [],
} satisfies Config;
